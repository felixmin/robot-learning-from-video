## https://github.com/MHVali/Noise-Substitution-in-Vector-Quantization/blob/main/NSVQ.py
## NSVQ: Noise Substitution in Vector Quantization for Machine Learning in IEEE Access journal, January 2022

import logging
import torch
import torch.distributions.normal as normal_dist
import torch.distributions.uniform as uniform_dist

logger = logging.getLogger(__name__)


## add project_in, project_out layer
## FYI vector_quantize_pytorch
class NSVQ(torch.nn.Module):
    def __init__(
        self,
        dim,
        num_embeddings,
        embedding_dim,
        device=torch.device("cpu"),
        discarding_threshold=0.1,
        initialization="normal",
        code_seq_len=1,
        patch_size=32,
        image_size=256,
        grid_size=None,
    ):
        super(NSVQ, self).__init__()

        """
        Inputs:
        
        1. num_embeddings = Number of codebook entries
        
        2. embedding_dim = Embedding dimension (dimensionality of each input data sample or codebook entry)
        
        3. device = The device which executes the code (CPU or GPU)
        
        ########## change the following inputs based on your application ##########
        
        4. discarding_threshold = Replacement cutoff as a fraction of average usage.
            When replacing unused entries, we compute the average usage per codebook
            entry over the replacement window and replace entries whose usage is
            below (discarding_threshold * average_usage).
        
        5. initialization = Initial distribution for codebooks
        
        6. grid_size = Explicit spatial grid size (h, w). If None, computed from image_size/patch_size

        """
        self.image_size = image_size
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.discarding_threshold = discarding_threshold
        self.eps = 1e-12
        self.dim = dim
        self.patch_size = patch_size

        # Compute effective grid size
        if grid_size is not None:
            self.grid_h, self.grid_w = grid_size
        else:
            self.grid_h = self.grid_w = int(image_size / patch_size)

        if initialization == "normal":
            codebooks = torch.randn(
                self.num_embeddings, self.embedding_dim, device=device
            )
        elif initialization == "uniform":
            codebooks = (
                uniform_dist.Uniform(-1 / self.num_embeddings, 1 / self.num_embeddings)
                .sample([self.num_embeddings, self.embedding_dim])
                .to(device)
            )
        else:
            raise ValueError(
                "initialization should be one of the 'normal' and 'uniform' strings"
            )

        self.codebooks = torch.nn.Parameter(codebooks, requires_grad=True)

        # Counter variable which contains the number of times each codebook is used.
        # Register as a buffer so it moves with the module across devices.
        self.register_buffer(
            "codebooks_used",
            torch.zeros(self.num_embeddings, dtype=torch.int32, device=device),
            persistent=False,
        )

        self.project_in = torch.nn.Linear(dim, embedding_dim)
        self.project_out = torch.nn.Linear(embedding_dim, dim)

        # Build CNN encoder based on grid size and target code_seq_len
        # Use predefined architectures for standard grid sizes, dynamic for others
        input_size = self.grid_h  # Assuming square grid

        if input_size == 8:
            # Original architectures for 8x8 grid (from image_size=256, patch_size=32)
            if code_seq_len == 1:
                self.cnn_encoder = torch.nn.Sequential(
                    torch.nn.Conv2d(
                        in_channels=embedding_dim,
                        out_channels=embedding_dim,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    ),
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(
                        in_channels=embedding_dim,
                        out_channels=embedding_dim,
                        kernel_size=4,
                        stride=1,
                        padding=0,
                    ),
                )
            elif code_seq_len == 2:
                self.cnn_encoder = torch.nn.Sequential(
                    torch.nn.Conv2d(
                        in_channels=embedding_dim,
                        out_channels=embedding_dim,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    ),
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(
                        in_channels=embedding_dim,
                        out_channels=embedding_dim,
                        kernel_size=(3, 4),
                        stride=1,
                        padding=0,
                    ),
                )
            elif code_seq_len == 4:
                self.cnn_encoder = torch.nn.Sequential(
                    torch.nn.Conv2d(
                        in_channels=embedding_dim,
                        out_channels=embedding_dim,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    ),
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(
                        in_channels=embedding_dim,
                        out_channels=embedding_dim,
                        kernel_size=3,
                        stride=1,
                        padding=0,
                    ),
                )
            else:
                raise ValueError(
                    f"code_seq_len={code_seq_len} not supported for 8x8 grid"
                )
        elif input_size == 16:
            # Architectures for 16x16 grid (from DINO with patch_size=16 on 256x256)
            if code_seq_len == 1:
                # 16 -> 8 -> 4 -> 2 -> 1
                self.cnn_encoder = torch.nn.Sequential(
                    torch.nn.Conv2d(
                        in_channels=embedding_dim,
                        out_channels=embedding_dim,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    ),
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(
                        in_channels=embedding_dim,
                        out_channels=embedding_dim,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    ),
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(
                        in_channels=embedding_dim,
                        out_channels=embedding_dim,
                        kernel_size=4,
                        stride=1,
                        padding=0,
                    ),
                )
            elif code_seq_len == 4:
                # 16 -> 8 -> 4 -> 2
                self.cnn_encoder = torch.nn.Sequential(
                    torch.nn.Conv2d(
                        in_channels=embedding_dim,
                        out_channels=embedding_dim,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    ),
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(
                        in_channels=embedding_dim,
                        out_channels=embedding_dim,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    ),
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(
                        in_channels=embedding_dim,
                        out_channels=embedding_dim,
                        kernel_size=3,
                        stride=1,
                        padding=0,
                    ),
                )
            elif code_seq_len == 16:
                # 16 -> 8 -> 4
                self.cnn_encoder = torch.nn.Sequential(
                    torch.nn.Conv2d(
                        in_channels=embedding_dim,
                        out_channels=embedding_dim,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    ),
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(
                        in_channels=embedding_dim,
                        out_channels=embedding_dim,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    ),
                )
            elif code_seq_len == 64:
                # 16 -> 8
                self.cnn_encoder = torch.nn.Sequential(
                    torch.nn.Conv2d(
                        in_channels=embedding_dim,
                        out_channels=embedding_dim,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    ),
                )
            else:
                raise ValueError(
                    f"code_seq_len={code_seq_len} not supported for 16x16 grid"
                )
        else:
            raise ValueError(
                f"Grid size {input_size}x{input_size} not supported. Use 8 or 16."
            )

    def encode(self, input_data, batch_size):
        # compute the distances between input and codebooks vectors
        input_data = self.project_in(input_data)  # b * 64 * 32
        # change the order of the input_data to b * 32 * 64
        input_data = input_data.permute(0, 2, 1).contiguous()
        # reshape input_data to 4D b*h*w*d using actual grid size
        input_data = input_data.reshape(
            batch_size, self.embedding_dim, self.grid_h, self.grid_w
        )
        input_data = self.cnn_encoder(input_data)  # 1*1 tensor
        input_data = input_data.reshape(
            batch_size, self.embedding_dim, -1
        )  # b * 32 * d^2
        input_data = input_data.permute(0, 2, 1).contiguous()  # b * 1 * 32
        input_data = input_data.reshape(-1, self.embedding_dim)
        return input_data

    def decode(self, quantized_input, batch_size):
        quantized_input = quantized_input.reshape(
            batch_size, self.embedding_dim, -1
        )  # b * 32 * d^2
        quantized_input = quantized_input.permute(0, 2, 1).contiguous()  # b * 64 * 32

        quantized_input = self.project_out(quantized_input)
        return quantized_input

    def forward(self, input_data_first, input_data_last, codebook_training_only=False):
        """
        This function performs the main proposed vector quantization function using NSVQ trick to pass the gradients.
        Use this forward function for training phase.

        N: number of input data samples
        K: num_embeddings (number of codebook entries)
        D: embedding_dim (dimensionality of each input data sample or codebook entry)

        input: input_data (input data matrix which is going to be vector quantized | shape: (NxD) )
        outputs:
                quantized_input (vector quantized version of input data used for training | shape: (NxD) )
                perplexity (average usage of codebook entries)
        """

        batch_size = input_data_first.shape[0]

        input_data_first = input_data_first.contiguous()
        input_data_last = input_data_last.contiguous()

        input_data_first = self.encode(input_data_first, batch_size)  # b * 1 * 32
        input_data_last = self.encode(input_data_last, batch_size)  # b * 1 * 32

        input_data = input_data_last - input_data_first

        distances = (
            torch.sum(input_data**2, dim=1, keepdim=True)
            - 2 * (torch.matmul(input_data, self.codebooks.t()))
            + torch.sum(self.codebooks.t() ** 2, dim=0, keepdim=True)
        )

        min_indices = torch.argmin(distances, dim=1)

        hard_quantized_input = self.codebooks[min_indices]

        # Use input_data.device to ensure random vector is on the correct device
        random_vector = torch.randn_like(input_data)

        norm_quantization_residual = (
            (input_data - hard_quantized_input).square().sum(dim=1, keepdim=True).sqrt()
        )
        norm_random_vector = random_vector.square().sum(dim=1, keepdim=True).sqrt()

        # defining vector quantization error
        vq_error = (
            norm_quantization_residual / norm_random_vector + self.eps
        ) * random_vector

        if codebook_training_only:
            logger.debug("codebook error: %s", norm_quantization_residual.norm())
            quantized_input = hard_quantized_input
        else:
            quantized_input = input_data + vq_error

        # calculating the perplexity (average usage of codebook entries)
        encodings = torch.zeros(
            input_data.shape[0], self.num_embeddings, device=input_data.device
        )
        encodings.scatter_(1, min_indices.reshape([-1, 1]), 1)
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + self.eps)))

        with torch.no_grad():
            # Correctly handle duplicate indices (advanced indexing would undercount).
            self.codebooks_used.scatter_add_(
                0,
                min_indices,
                torch.ones_like(min_indices, dtype=self.codebooks_used.dtype),
            )

        # use the first returned tensor "quantized_input" for training phase (Notice that you do not have to use the
        # tensor "quantized_input" for inference (evaluation) phase)
        # Also notice you do not need to add a new loss term (for VQ) to your global loss function to optimize codebooks.
        # Just return the tensor of "quantized_input" as vector quantized version of the input data.

        quantized_input = self.decode(quantized_input, batch_size)
        # NOTE: Avoid unconditional GPU->CPU sync/transfer in the hot path.
        # Callers that need CPU/numpy stats should explicitly request/convert them.
        return (
            quantized_input,
            perplexity,
            self.codebooks_used,
            min_indices.reshape(batch_size, -1),
        )

    def _is_distributed(self) -> bool:
        try:
            import torch.distributed as dist
        except Exception:
            return False
        return dist.is_available() and dist.is_initialized()

    def _get_replacement_indices_from_counts(
        self,
        counts: torch.Tensor,
        discarding_threshold: float | None = None,
    ):
        """
        Compute which codebook entries are considered unused/used for replacement.

        Replacement criterion (size-aware):
          - Let `total = sum(codebooks_used)` within the current replacement window.
          - Let `expected = total / num_embeddings` (average usage per entry).
          - Mark entry i as "unused" if `codebooks_used[i] < discarding_threshold * expected`.

        This scales naturally with codebook size: for large codebooks, the average usage
        per entry is lower, so the cutoff is lower as well.

        Returns:
            unused_indices: 1D LongTensor of indices to replace
            used_indices: 1D LongTensor of indices considered active
            min_count: Float cutoff used for the split
        """
        total = int(counts.sum().item())

        if total <= 0 or self.num_embeddings <= 0:
            # No information in the window; treat everything as unused.
            all_idx = torch.arange(
                self.num_embeddings, dtype=torch.long, device=counts.device
            )
            empty = torch.empty((0,), dtype=torch.long, device=counts.device)
            return all_idx, empty, 0.0

        expected = total / float(self.num_embeddings)
        threshold = (
            float(self.discarding_threshold)
            if discarding_threshold is None
            else float(discarding_threshold)
        )
        min_count = threshold * expected

        counts_f = counts.to(dtype=torch.float32)
        used_indices = torch.where(counts_f >= min_count)[0].to(dtype=torch.long)
        unused_indices = torch.where(counts_f < min_count)[0].to(dtype=torch.long)

        return unused_indices, used_indices, min_count

    def _get_replacement_indices(self, discarding_threshold: float | None = None):
        return self._get_replacement_indices_from_counts(
            self.codebooks_used,
            discarding_threshold=discarding_threshold,
        )

    def replace_unused_codebooks(self, discarding_threshold: float | None = None):
        """
        Replace inactive codebook entries with (noisy) copies of active ones.

        The replacement criterion is size-aware:
          - Let total = total assignments in the replacement window.
          - Let expected = total / num_embeddings (average usage per entry).
          - Replace entries with usage < discarding_threshold * expected.

        """

        with torch.no_grad():
            # Compute replacement decision on globally aggregated usage in DDP, so we don't
            # accidentally replace entries that are used on other ranks.
            counts = self.codebooks_used.to(dtype=torch.int64)
            if self._is_distributed():
                import torch.distributed as dist

                counts = counts.clone()
                dist.all_reduce(counts, op=dist.ReduceOp.SUM)

            total_assignments = int(counts.sum().item())
            threshold = (
                float(self.discarding_threshold)
                if discarding_threshold is None
                else float(discarding_threshold)
            )
            unused_indices, used_indices, min_count = (
                self._get_replacement_indices_from_counts(
                    counts,
                    discarding_threshold=threshold,
                )
            )

            unused_count = unused_indices.shape[0]
            used_count = used_indices.shape[0]

            # Safety invariants: disjoint partition of the full codebook.
            if int(unused_count + used_count) != int(self.num_embeddings):
                raise RuntimeError(
                    f"Replacement index partition mismatch: unused={int(unused_count)} "
                    f"used={int(used_count)} num_embeddings={int(self.num_embeddings)}"
                )
            if unused_count > 0 and used_count > 0:
                used_mask = torch.zeros(
                    self.num_embeddings, dtype=torch.bool, device=unused_indices.device
                )
                used_mask[used_indices] = True
                if bool(used_mask[unused_indices].any().item()):
                    raise RuntimeError(
                        "Replacement indices overlap: some indices are both used and unused"
                    )

            is_dist = self._is_distributed()
            dist_rank = 0
            if is_dist:
                import torch.distributed as dist

                dist_rank = int(dist.get_rank())

            if used_count == 0:
                if dist_rank == 0:
                    logger.info("No active codebooks; shuffling entire codebook")
                    self.codebooks += (
                        self.eps
                        * torch.randn(
                            self.codebooks.size(), device=self.codebooks.device
                        ).clone()
                    )
            elif unused_count == 0:
                # Everything is considered active in this window; nothing to replace.
                pass
            else:
                if dist_rank == 0:
                    used = self.codebooks[used_indices].clone()
                    if used_count < unused_count:
                        used_codebooks = used.repeat(
                            int((unused_count / (used_count + self.eps)) + 1), 1
                        )
                        used_codebooks = used_codebooks[
                            torch.randperm(
                                used_codebooks.shape[0], device=used_codebooks.device
                            )
                        ]
                    else:
                        used_codebooks = used

                    self.codebooks[unused_indices] = used_codebooks[
                        :unused_count
                    ] + 0.02 * torch.randn(
                        (unused_count, self.embedding_dim), device=self.codebooks.device
                    )

            # Broadcast updated codebooks so all ranks stay in sync.
            if is_dist:
                import torch.distributed as dist

                dist.broadcast(self.codebooks.data, src=0)

            logger.info(
                "Replaced %d codebooks (used=%d, total=%d, min_count=%.4f, threshold=%.6f)",
                unused_count,
                used_count,
                total_assignments,
                min_count,
                threshold,
            )
            self.codebooks_used.zero_()
            return (
                int(unused_count),
                int(used_count),
                int(total_assignments),
                float(min_count),
            )

    def get_indices(
        self, input_data_first: torch.Tensor, input_data_last: torch.Tensor
    ) -> torch.Tensor:
        """Return quantization indices without updating the codebooks_used counter."""
        batch_size = input_data_first.shape[0]
        first = self.encode(input_data_first.contiguous(), batch_size)
        last = self.encode(input_data_last.contiguous(), batch_size)
        delta = last - first
        distances = (
            torch.sum(delta**2, dim=1, keepdim=True)
            - 2 * torch.matmul(delta, self.codebooks.t())
            + torch.sum(self.codebooks.t() ** 2, dim=0, keepdim=True)
        )
        return torch.argmin(distances, dim=1).reshape(batch_size, -1)

    def inference(self, input_data_first, input_data_last, user_action_token_num=None):
        """
        This function performs the vector quantization function for inference (evaluation) time (after training).
        This function should not be used during training.

        N: number of input data samples
        K: num_embeddings (number of codebook entries)
        D: embedding_dim (dimensionality of each input data sample or codebook entry)

        input: input_data (input data matrix which is going to be vector quantized | shape: (NxD) )
        outputs:
                quantized_input (vector quantized version of input data used for inference (evaluation) | shape: (NxD) )
        """

        input_data_first = input_data_first.detach().clone()
        input_data_last = input_data_last.detach().clone()
        codebooks = self.codebooks.detach().clone()

        batch_size = input_data_first.shape[0]
        # compute the distances between input and codebooks vectors

        input_data_first = self.encode(input_data_first, batch_size)  # b * n * dim
        input_data_last = self.encode(input_data_last, batch_size)  # b * n * dim

        input_data = input_data_last - input_data_first

        input_data = input_data.reshape(-1, self.embedding_dim)

        distances = (
            torch.sum(input_data**2, dim=1, keepdim=True)
            - 2 * (torch.matmul(input_data, codebooks.t()))
            + torch.sum(codebooks.t() ** 2, dim=0, keepdim=True)
        )

        min_indices = torch.argmin(distances, dim=1)

        if user_action_token_num is not None:
            if isinstance(user_action_token_num, list):
                min_indices = torch.tensor(
                    user_action_token_num, device=input_data.device
                )
            else:
                min_indices = torch.tensor(
                    [[user_action_token_num]], device=input_data.device
                ).repeat(input_data.shape[0], 1)
        quantized_input = codebooks[min_indices]

        quantized_input = self.decode(quantized_input, batch_size)

        # use the tensor "quantized_input" as vector quantized version of your input data for inference (evaluation) phase.
        return quantized_input, min_indices.reshape(batch_size, -1)

    def codebook_reinit(self):
        self.codebooks = torch.nn.Parameter(
            torch.randn(
                self.num_embeddings, self.embedding_dim, device=self.codebooks.device
            ),
            requires_grad=True,
        )
        self.codebooks_used.zero_()

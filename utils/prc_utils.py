import torch
import galois
import numpy as np
from scipy.sparse import csr_matrix
from scipy.special import binom
from ldpc import BpDecoder
from tqdm import tqdm


GF = galois.GF(2)


def pseudorandom_code_key_gen(
        codeword_len, 
        message_len, 
        false_positive_rate=0.01, 
        t=3, 
        g=None, 
        r=None, 
        noise_rate=None,
        max_bp_iter=None,
    ):
    """
    Generate the encoding and decoding keys for pseudorandom error-correction code.
    Proposed by Miranda Christ and Sam Gunn. Pseudorandom error-correcting codes. 
    In Annual International Cryptology Conference, pages 325–347. Springer, 2024.

    Parameters
    ----------
    codeword_len : int
        The length of the PRC codeword
    message_len : int
        The length of the message you want to encode
    false_positive_rate : float, optional
        The false positive rate you're willing to tolerate, by default 0.01
    t : int, optional
        The sparsity of parity checks. Larger values help pseudorandomness, by default 3
    g : int, optional
        The dimension of random code used. Larger values help pseudorandomness
    r : int, optional
        The number of parity checks used. Smaller values help pseudorandomness
    noise_rate : float, optional
        The amount of noise for Encode to add to codewords. Larger values help pseudorandomness
    max_bp_iter: int, optional
        The maximum number of BP iterations, larger values help noise robustness

    Returns
    -------
    encoding_key : (generator_matrix, one_time_pad, test_bits, g, noise_rate)
        The encoding key
    decoding_key : (generator_matrix, parity_check_matrix, one_time_pad, false_positive_rate,
                        noise_rate, test_bits, g, max_bp_iter, t)
        The decoding key
            
    """
    # Set basic scheme parameters
    num_test_bits = int(np.ceil(np.log2(1 / false_positive_rate)))
    secpar = int(np.log2(binom(codeword_len, t)))
    g = secpar if g is None else g
    noise_rate = 1 - 2 ** (-secpar / g ** 2) if noise_rate is None else noise_rate
    k = message_len + g + num_test_bits
    r = codeword_len - k - secpar if r is None else r
    # Sample n by k generator matrix (all but the first n-r of these will be over-written)
    generator_matrix = GF.Random((codeword_len, k)) 
    # Sample scipy.sparse parity-check matrix together with the last n-r rows of the generator matrix
    row_indices, col_indices, data = [], [], []
    for row in tqdm(range(r), desc=f"Generating PRC key for codeword length {codeword_len} and message length {message_len}"):
        chosen_indices = np.random.choice(codeword_len - r + row, t - 1, replace=False)
        chosen_indices = np.append(chosen_indices, codeword_len - r + row)
        row_indices.extend([row] * t)
        col_indices.extend(chosen_indices)
        data.extend([1] * t)
        generator_matrix[codeword_len - r + row] = generator_matrix[chosen_indices[:-1]].sum(axis=0)
    parity_check_matrix = csr_matrix((data, (row_indices, col_indices)))
    # Compute scheme parameters
    max_bp_iter = int(np.log(codeword_len) / np.log(t)) if max_bp_iter is None else max_bp_iter
    # Sample one-time pad and test bits
    one_time_pad = GF.Random(codeword_len)
    test_bits = GF.Random(num_test_bits)
    # Permute bits
    permutation = np.random.permutation(codeword_len)
    generator_matrix = generator_matrix[permutation]
    one_time_pad = one_time_pad[permutation]
    parity_check_matrix = parity_check_matrix[:, permutation]
    encoding_key = (generator_matrix, one_time_pad, test_bits, g, noise_rate)
    decoding_key = (generator_matrix, parity_check_matrix, one_time_pad, false_positive_rate, \
                        noise_rate, test_bits, g, max_bp_iter, t)
    return encoding_key, decoding_key


def pseudorandom_code_encode(message, encoding_key):
    """
    Encode a message using the pseudorandom code scheme.
    Proposed by Miranda Christ and Sam Gunn. Pseudorandom error-correcting codes. 
    In Annual International Cryptology Conference, pages 325–347. Springer, 2024.

    Parameters
    ----------
    message : np.ndarray, shape (message_len,)
        The message to be encoded.
    encoding_key : tuple, (generator_matrix, one_time_pad, test_bits, g, noise_rate)
        The encoding key containing the generator matrix, one-time pad, test bits, g, and noise rate.

    Returns
    -------
    prc_codeword : np.ndarray, shape (codeword_len,)
        The PRC codeword.
    """
    generator_matrix, one_time_pad, test_bits, g, noise_rate = encoding_key
    n, k = generator_matrix.shape
    assert len(message) <= k - len(test_bits) - g, "Message is too long"
    payload = np.concatenate((test_bits, GF.Random(g), GF(message), GF.Zeros(k- len(test_bits) - g - len(message))))
    error = GF(np.random.binomial(1, noise_rate, n))
    prc_codeword = 1 - 2 * torch.tensor(payload @ generator_matrix.T + one_time_pad + error, dtype=torch.float).numpy()
    return prc_codeword


def pseudorandom_code_detect(posteriors, decoding_key, false_positive_rate=None):
    """
    Detect whether a codeword matches the given key under PRC decoding.
    Proposed by Miranda Christ and Sam Gunn. Pseudorandom error-correcting codes. 
    In Annual International Cryptology Conference, pages 325–347. Springer, 2024.

    Parameters
    ----------
    posteriors : np.ndarray, shape (codeword_len,)
        The posteriors to be detected.
    decoding_key : tuple, (generator_matrix, parity_check_matrix, one_time_pad, false_positive_rate,
                            noise_rate, test_bits, g, max_bp_iter, t)
        The decoding key containing the generator matrix, parity check matrix, one-time pad, false positive rate,
        noise rate, test bits, g, maximum number of belief propagation iterations, and t.
    false_positive_rate : float
        The false positive rate of the detector.

    Returns
    -------
    detected : bool
        True if the codeword matches the key, False otherwise.
    """
    (generator_matrix, parity_check_matrix, one_time_pad, false_positive_rate_key, \
                        noise_rate, test_bits, g, max_bp_iter, t) = decoding_key
    false_positive_rate = false_positive_rate_key if false_positive_rate is None else false_positive_rate
    posteriors = (1 - 2 * noise_rate) * (1 - 2 * np.array(one_time_pad, dtype=float)) * posteriors
    r = parity_check_matrix.shape[0]
    Pi = np.prod(posteriors[parity_check_matrix.indices.reshape(r, t)], axis=1)
    log_plus = np.log((1 + Pi) / 2)
    log_minus = np.log((1 - Pi) / 2)
    log_prod = log_plus + log_minus
    const = 0.5 * np.sum(np.power(log_plus, 2) + np.power(log_minus, 2) - 0.5 * np.power(log_prod, 2))
    threshold = np.sqrt(2 * const * np.log(1 / false_positive_rate)) + 0.5 * log_prod.sum()
    detected = log_plus.sum() >= threshold
    return detected, log_plus.sum()


def _boolean_row_reduce(A: np.ndarray):
    A_bool = np.asarray(A, dtype=bool)
    if not A_bool.flags.c_contiguous:
        A_bool = np.ascontiguousarray(A_bool)
    n, k = A_bool.shape
    pad = (-k) % 8
    if pad:
        A_bool = np.pad(A_bool, ((0,0),(0,pad)), constant_values=False)
    A_packed = np.packbits(A_bool, axis=1)  # uint8
    nb = A_packed.shape[1]
    perm = np.arange(n)
    r = 0
    for j in range(k):
        byte_idx, bit_off = divmod(j, 8)
        mask = (A_packed[r:, byte_idx] >> (7 - bit_off)) & 1
        if not mask.any():
            return None
        i = r + mask.argmax()
        if i != r:
            A_packed[[r, i]] = A_packed[[i, r]]
            perm[[r, i]] = perm[[i, r]]
        below = ((A_packed[r+1:, byte_idx] >> (7 - bit_off)) & 1).astype(bool)
        if below.any():
            A_packed[r+1:, :] ^= A_packed[r] * below[:, None]
        r += 1
        if r == n:
            break
    return perm[:k]


def pseudorandom_code_decode(posteriors, decoding_key):
    """
    Decode a PRC codeword using the pseudorandom code scheme.
    Proposed by Miranda Christ and Sam Gunn. Pseudorandom error-correcting codes. 
    In Annual International Cryptology Conference, pages 325–347. Springer, 2024.

    Parameters
    ----------
    posteriors : np.ndarray, shape (codeword_len,)
        The posteriors to be decoded.
    decoding_key : tuple, (generator_matrix, parity_check_matrix, one_time_pad, false_positive_rate,
                            noise_rate, test_bits, g, max_bp_iter, t)

    Returns
    -------
    message : np.ndarray, shape (message_len,)
        The decoded message.
    """
    (generator_matrix, parity_check_matrix, one_time_pad, false_positive_rate, \
                        noise_rate, test_bits, g, max_bp_iter, t) = decoding_key
    posteriors = (1 - 2 * noise_rate) * (1 - 2 * np.array(one_time_pad, dtype=float)) * posteriors
    channel_probs = (1 - np.abs(posteriors)) / 2
    x_recovered = ((1 - np.sign(posteriors)) // 2).astype(int)
    # Apply the belief-propagation decoder.
    # bpd = bp_decoder(parity_check_matrix, channel_probs=channel_probs, max_iter=max_bp_iter, bp_method="product_sum")
    bpd = BpDecoder(parity_check_matrix, channel_probs=channel_probs, max_iter=max_bp_iter, bp_method="product_sum")
    x_decoded = bpd.decode(x_recovered)
    # Compute a confidence score.
    bpd_probs = 1 / (1 + np.exp(bpd.log_prob_ratios))
    confidences = 2 * np.abs(0.5 - bpd_probs)
    # Order codeword bits by confidence.
    confidence_order = np.argsort(-confidences)
    ordered_generator_matrix = generator_matrix[confidence_order]
    ordered_x_decoded = x_decoded[confidence_order]
    # Find the first (according to the confidence order) linearly independent set of rows of the generator matrix.
    top_invertible_rows = _boolean_row_reduce(ordered_generator_matrix)
    if top_invertible_rows is None:
        return None
    # Solve the system.
    recovered_string = np.linalg.solve(ordered_generator_matrix[top_invertible_rows], GF(ordered_x_decoded[top_invertible_rows]))
    message = recovered_string[len(test_bits) + g:]
    return np.array(message)

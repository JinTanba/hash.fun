import numpy as np
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import hamming

# Load a pre-trained Sentence-BERT model
model = SentenceTransformer('all-MiniLM-L6-v2')

def compute_embedding(text):
    """
    Compute the continuous embedding for a given text using Sentence-BERT.
    """
    encoded = model.encode(text)
    print(">>>> compute_embedding >>>>>",encoded)
    return encoded

def binarize_embedding(embedding, threshold=0.0):
    """
    Binarize the embedding vector by thresholding.
    
    Parameters:
    - embedding: a numpy array of floats.
    - threshold: a threshold value. Dimensions with values greater than the threshold are set to 1; otherwise 0.
    
    Returns:
    - A binary numpy array (vector) representing the hash code.
    """
    binary_code = (embedding > threshold).astype(int)
    print(">>>> binarize_embedding >>>>>",binary_code)
    return binary_code

def compute_hash(text, threshold=0.0):
    """
    Generate a binary hash code for a given text.
    
    Steps:
      1. Compute the continuous embedding for the text.
      2. Binarize the embedding using the specified threshold.
    """
    embedding = compute_embedding(text)
    return binarize_embedding(embedding, threshold)

def compute_hamming_distance(hash1, hash2):
    """
    Compute the Hamming distance between two binary hash codes.
    
    Here, we use scipy's hamming function which returns the normalized Hamming distance.
    Multiply by the code length to get the count of differing bits.
    """
    # Ensure both codes are numpy arrays
    hash1 = np.array(hash1)
    hash2 = np.array(hash2)
    # hamming() returns a normalized value; multiply by length to get number of differing bits.
    return hamming(hash1, hash2) * len(hash1)

def jaccard_similarity(hash1, hash2):
    """
    Optionally, compute the Jaccard similarity between two binary hash codes.
    
    Jaccard similarity = (Intersection size) / (Union size)
    """
    hash1 = np.array(hash1)
    hash2 = np.array(hash2)
    intersection = np.sum(np.logical_and(hash1, hash2))
    union = np.sum(np.logical_or(hash1, hash2))
    return intersection / union if union > 0 else 1.0

def hash_to_str(binary_hash):
    """
    Convert a binary hash (a NumPy array of 0's and 1's) into a hexadecimal string.
    
    Parameters:
        binary_hash (np.ndarray): The binary hash code, e.g., output of binarize_embedding.
        
    Returns:
        A hexadecimal string representation of the binary hash.
    """
    # Ensure the hash is of type uint8
    binary_hash = np.array(binary_hash, dtype=np.uint8)
    # Pack bits into bytes; note that the length must be a multiple of 8.
    # If it’s not, you may want to pad the array.
    packed = np.packbits(binary_hash)
    # Convert the packed bytes to a hex string
    hex_str = packed.tobytes().hex()
    return hex_str

def str_to_hash(hex_str, original_length=None):
    """
    Convert a hexadecimal string back into the binary hash (NumPy array of 0's and 1's).
    
    Parameters:
        hex_str (str): The hexadecimal string representation of the binary hash.
        original_length (int, optional): The length of the original binary hash.
            If provided, the output array will be trimmed to that length.
        
    Returns:
        A NumPy array of 0's and 1's representing the binary hash.
    """
    # Convert hex string back to bytes
    byte_data = bytes.fromhex(hex_str)
    # Convert bytes into a NumPy array of uint8, then unpack bits into a binary array
    bits = np.unpackbits(np.frombuffer(byte_data, dtype=np.uint8))
    # If an original length was provided, trim the extra bits (if any)
    if original_length is not None:
        bits = bits[:original_length]
    return bits


# Example usage
if __name__ == '__main__':
    text1 = "The quick brown fox jumps over the lazy dog.A fast, dark-colored fox leaps above a sleepy canine.A fast, dark-colored fox leaps above a sleepy canine.A fast, dark-colored fox leaps above a sleepy canineThe quick brown fox jumps over the lazy dog.A fast, dark-colored fox leaps above a sleepy canine.A fast, dark-colored fox leaps above a sleepy canine.A fast, dark-colored fox leaps above a sleepy canine."
    text2 = "A fast, dark-colored fox leaps above a sleepy canine."
    text3 = "The stock market experienced a significant crash yesterday."
    print("start-------")
    # Compute hash codes for each text
    hash1 = compute_hash(text1)
    hash1InHash = hash_to_str(hash1)
    print(hash1InHash)

    hash2 = compute_hash(text2)
    # print(hash2)
    hash3 = compute_hash(text3)
    # print(hash3)
    hash1 = str_to_hash(hash1)
    # Compute similarities
    hamming_dist_1_2 = compute_hamming_distance(hash1, hash2)
    hamming_dist_1_3 = compute_hamming_distance(hash1, hash3)

    jaccard_sim_1_2 = jaccard_similarity(hash1, hash2)
    jaccard_sim_1_3 = jaccard_similarity(hash1, hash3)

    print("Hamming distance between text1 and text2:", hamming_dist_1_2)
    print("Hamming distance between text1 and text3:", hamming_dist_1_3)
    print("Jaccard similarity between text1 and text2:", jaccard_sim_1_2)
    print("Jaccard similarity between text1 and text3:", jaccard_sim_1_3)
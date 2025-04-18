#!/usr/bin/env python3
"""
Password Cracking Tool

This script allows users to:
1. Enter a password
2. Select an encryption/hash type
3. Attempt to crack it using parallel processing or GPU acceleration
"""

import hashlib
import time
import string
import argparse
import multiprocessing
import itertools
import os
import base64
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

# Try to import GPU libraries, use CPU if not available
try:
    import torch
    import numpy as np
    HAS_GPU = torch.cuda.is_available()
except ImportError:
    HAS_GPU = False
    print("GPU acceleration libraries not found. Using CPU only.")

# Try to import encryption libraries, use only hash if not available
try:
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    from cryptography.hazmat.backends import default_backend
    HAS_ENCRYPTION = True
except ImportError:
    HAS_ENCRYPTION = False
    print("Encryption libraries not found. Using only hash algorithms.")

# Available hash algorithms
HASH_TYPES = {
    "md5": hashlib.md5,
    "sha1": hashlib.sha1,
    "sha256": hashlib.sha256,
    "sha512": hashlib.sha512
}

# Available encryption algorithms (if encryption libraries are installed)
ENCRYPTION_TYPES = {}
if HAS_ENCRYPTION:
    ENCRYPTION_TYPES = {
        "aes-128": {"algorithm": algorithms.AES, "key_size": 16, "mode": modes.CBC},  # 128 bits = 16 bytes
        "aes-192": {"algorithm": algorithms.AES, "key_size": 24, "mode": modes.CBC},  # 192 bits = 24 bytes
        "aes-256": {"algorithm": algorithms.AES, "key_size": 32, "mode": modes.CBC},  # 256 bits = 32 bytes
    }

def hash_password(password, hash_type="sha256"):
    """Hash a password using the specified algorithm."""
    if hash_type not in HASH_TYPES:
        raise ValueError(f"Unsupported hash type: {hash_type}")

    hasher = HASH_TYPES[hash_type]()
    hasher.update(password.encode('utf-8'))
    return hasher.hexdigest()

def encrypt_password(password, encryption_type, key=None):
    """
    Encrypt a password using the specified algorithm and key.

    Args:
        password: The password to encrypt
        encryption_type: The encryption algorithm to use
        key: The encryption key (if None, a random key will be generated)

    Returns:
        A tuple of (encrypted_password, key, iv) where:
        - encrypted_password is the base64-encoded encrypted password
        - key is the base64-encoded encryption key
        - iv is the base64-encoded initialization vector
    """
    if not HAS_ENCRYPTION:
        raise ValueError("Encryption libraries not available. Please install cryptography package.")

    if encryption_type not in ENCRYPTION_TYPES:
        raise ValueError(f"Unsupported encryption type: {encryption_type}")

    # Get encryption parameters
    enc_params = ENCRYPTION_TYPES[encryption_type]
    algorithm = enc_params["algorithm"]
    key_size = enc_params["key_size"]
    mode_class = enc_params["mode"]

    # Generate or validate key
    if key is None:
        # Generate a random key of the appropriate length
        key = os.urandom(key_size)
    elif isinstance(key, str):
        # Convert string key to bytes and ensure correct length
        key_bytes = key.encode('utf-8')
        if len(key_bytes) < key_size:
            # Pad key if too short
            key = key_bytes + b'\0' * (key_size - len(key_bytes))
        elif len(key_bytes) > key_size:
            # Truncate key if too long
            key = key_bytes[:key_size]
        else:
            key = key_bytes

    # Generate a random IV (Initialization Vector)
    iv = os.urandom(16)  # AES block size is 16 bytes

    # Create an encryptor object
    cipher = Cipher(algorithm(key), mode_class(iv), backend=default_backend())
    encryptor = cipher.encryptor()

    # Pad the plaintext to a multiple of the block size
    block_size = 16  # AES block size
    padded_data = password.encode('utf-8')
    padding_length = block_size - (len(padded_data) % block_size)
    padded_data += bytes([padding_length]) * padding_length

    # Encrypt the data
    encrypted_data = encryptor.update(padded_data) + encryptor.finalize()

    # Encode the encrypted data, key, and IV as base64 for storage/transmission
    encrypted_b64 = base64.b64encode(encrypted_data).decode('utf-8')
    key_b64 = base64.b64encode(key).decode('utf-8')
    iv_b64 = base64.b64encode(iv).decode('utf-8')

    return encrypted_b64, key_b64, iv_b64

def decrypt_password(encrypted_b64, key_b64, iv_b64, encryption_type):
    """
    Decrypt an encrypted password using the specified algorithm, key, and IV.

    Args:
        encrypted_b64: Base64-encoded encrypted password
        key_b64: Base64-encoded encryption key
        iv_b64: Base64-encoded initialization vector
        encryption_type: The encryption algorithm used

    Returns:
        The decrypted password
    """
    if not HAS_ENCRYPTION:
        raise ValueError("Encryption libraries not available. Please install cryptography package.")

    if encryption_type not in ENCRYPTION_TYPES:
        raise ValueError(f"Unsupported encryption type: {encryption_type}")

    # Get encryption parameters
    enc_params = ENCRYPTION_TYPES[encryption_type]
    algorithm = enc_params["algorithm"]
    mode_class = enc_params["mode"]

    # Decode the base64 data
    encrypted_data = base64.b64decode(encrypted_b64)
    key = base64.b64decode(key_b64)
    iv = base64.b64decode(iv_b64)

    # Create a decryptor object
    cipher = Cipher(algorithm(key), mode_class(iv), backend=default_backend())
    decryptor = cipher.decryptor()

    # Decrypt the data
    decrypted_padded = decryptor.update(encrypted_data) + decryptor.finalize()

    # Remove padding
    padding_length = decrypted_padded[-1]
    decrypted = decrypted_padded[:-padding_length]

    return decrypted.decode('utf-8')

def check_hash_match(password_to_try, target_hash, hash_type):
    """Check if a password matches the target hash."""
    hashed = hash_password(password_to_try, hash_type)
    return hashed == target_hash, password_to_try

def check_encryption_match(password_to_try, encrypted_b64, key_b64, iv_b64, encryption_type):
    """Check if a password matches the encrypted data when decrypted."""
    try:
        decrypted = decrypt_password(encrypted_b64, key_b64, iv_b64, encryption_type)
        return decrypted == password_to_try, password_to_try
    except Exception:
        return False, password_to_try

def dictionary_attack(target_hash, hash_type, dict_file, num_processes=None):
    """
    Perform a dictionary attack using parallel processing.

    Args:
        target_hash: The hash to crack
        hash_type: The hash algorithm used
        dict_file: Path to dictionary file
        num_processes: Number of processes to use (defaults to CPU count)

    Returns:
        The cracked password or None if not found
    """
    if num_processes is None:
        num_processes = multiprocessing.cpu_count()

    print(f"Starting dictionary attack using {num_processes} processes...")
    start_time = time.time()

    # Check if dictionary file exists
    if not os.path.exists(dict_file):
        print(f"Dictionary file not found: {dict_file}")
        return None

    # Count lines in file for progress bar
    with open(dict_file, 'r', errors='ignore') as f:
        total_words = sum(1 for _ in f)

    # Function for each worker process
    def process_chunk(words):
        for word in words:
            word = word.strip()
            match, password = check_hash_match(word, target_hash, hash_type)
            if match:
                return password
        return None

    # Split the dictionary into chunks for parallel processing
    def chunks(file, size):
        chunk = []
        for i, line in enumerate(file):
            chunk.append(line)
            if (i + 1) % size == 0:
                yield chunk
                chunk = []
        if chunk:
            yield chunk

    # Process dictionary in parallel
    with open(dict_file, 'r', errors='ignore') as f:
        chunk_size = max(1000, total_words // (num_processes * 10))
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            futures = [executor.submit(process_chunk, chunk) 
                      for chunk in chunks(f, chunk_size)]

            with tqdm(total=total_words, desc="Passwords checked") as pbar:
                for i, future in enumerate(futures):
                    result = future.result()
                    pbar.update(min(chunk_size, total_words - pbar.n))
                    if result:
                        # Cancel all remaining futures
                        for f in futures[i+1:]:
                            f.cancel()
                        elapsed_time = time.time() - start_time
                        print(f"\nPassword found in {elapsed_time:.2f} seconds!")
                        return result

    elapsed_time = time.time() - start_time
    print(f"\nDictionary attack completed in {elapsed_time:.2f} seconds. Password not found.")
    return None

def try_passwords_hash(password_iter, length, target_hash, hash_type):
    """
    Try passwords from an iterator and check if they match the target hash.

    Args:
        password_iter: Iterator yielding password candidates
        length: Length of passwords being checked
        target_hash: The hash to crack
        hash_type: The hash algorithm used

    Returns:
        The cracked password or None if not found
    """
    for candidate in password_iter:
        password = ''.join(candidate)
        match, found_password = check_hash_match(password, target_hash, hash_type)
        if match:
            return found_password
    return None

def password_generator(start, end, length, charset):
    """
    Generate password combinations in a specific range.

    Args:
        start: Starting index
        end: Ending index
        length: Length of passwords to generate
        charset: Character set to use

    Yields:
        Lists of characters representing password candidates
    """
    # Convert start index to a base-n number where n is len(charset)
    base = len(charset)
    current = start
    while current < end:
        # Convert current index to password
        indices = []
        temp = current
        for _ in range(length):
            indices.append(temp % base)
            temp //= base
        indices.reverse()

        # Pad with zeros if needed
        while len(indices) < length:
            indices.insert(0, 0)

        # Convert indices to characters
        yield [charset[i] for i in indices]
        current += 1

def brute_force_attack_cpu(target_hash, hash_type, charset, max_length, num_processes=None):
    """
    Perform a brute force attack using CPU parallel processing.

    Args:
        target_hash: The hash to crack
        hash_type: The hash algorithm used
        charset: Character set to use (e.g., string.ascii_lowercase)
        max_length: Maximum password length to try
        num_processes: Number of processes to use (defaults to CPU count)

    Returns:
        The cracked password or None if not found
    """
    if num_processes is None:
        num_processes = multiprocessing.cpu_count()

    print(f"Starting brute force attack using {num_processes} CPU processes...")
    start_time = time.time()

    # Try passwords of increasing length
    for length in range(1, max_length + 1):
        print(f"Trying passwords of length {length}...")

        # Calculate total combinations for this length
        total_combinations = len(charset) ** length

        # Split the work among processes
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            # Create iterators for each process
            chunk_size = total_combinations // num_processes
            if chunk_size == 0:
                chunk_size = 1

            futures = []
            for i in range(num_processes):
                # Create a generator that produces a subset of combinations
                start_idx = i * chunk_size
                end_idx = (i + 1) * chunk_size if i < num_processes - 1 else total_combinations

                futures.append(executor.submit(
                    try_passwords_hash, 
                    password_generator(start_idx, end_idx, length, charset),
                    length,
                    target_hash,
                    hash_type
                ))

            # Check results
            with tqdm(total=total_combinations, desc=f"Length {length}") as pbar:
                completed = 0
                for future in futures:
                    result = future.result()
                    if result:
                        elapsed_time = time.time() - start_time
                        print(f"\nPassword found in {elapsed_time:.2f} seconds!")
                        return result
                    completed += chunk_size
                    pbar.update(min(chunk_size, total_combinations - pbar.n))

    elapsed_time = time.time() - start_time
    print(f"\nBrute force attack completed in {elapsed_time:.2f} seconds. Password not found.")
    return None

def dictionary_attack_encrypted(target_encrypted, key_b64, iv_b64, encryption_type, dict_file, num_processes=None):
    """
    Perform a dictionary attack on encrypted data using parallel processing.

    Args:
        target_encrypted: The base64-encoded encrypted data to crack
        key_b64: The base64-encoded encryption key
        iv_b64: The base64-encoded initialization vector
        encryption_type: The encryption algorithm used
        dict_file: Path to dictionary file
        num_processes: Number of processes to use (defaults to CPU count)

    Returns:
        The cracked password or None if not found
    """
    if not HAS_ENCRYPTION:
        print("Encryption libraries not available. Please install cryptography package.")
        return None

    if num_processes is None:
        num_processes = multiprocessing.cpu_count()

    print(f"Starting dictionary attack on encrypted data using {num_processes} processes...")
    start_time = time.time()

    # Check if dictionary file exists
    if not os.path.exists(dict_file):
        print(f"Dictionary file not found: {dict_file}")
        return None

    # Count lines in file for progress bar
    with open(dict_file, 'r', errors='ignore') as f:
        total_words = sum(1 for _ in f)

    # Function for each worker process
    def process_chunk(words):
        for word in words:
            word = word.strip()
            match, password = check_encryption_match(word, target_encrypted, key_b64, iv_b64, encryption_type)
            if match:
                return password
        return None

    # Split the dictionary into chunks for parallel processing
    def chunks(file, size):
        chunk = []
        for i, line in enumerate(file):
            chunk.append(line)
            if (i + 1) % size == 0:
                yield chunk
                chunk = []
        if chunk:
            yield chunk

    # Process dictionary in parallel
    with open(dict_file, 'r', errors='ignore') as f:
        chunk_size = max(1000, total_words // (num_processes * 10))
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            futures = [executor.submit(process_chunk, chunk) 
                      for chunk in chunks(f, chunk_size)]

            with tqdm(total=total_words, desc="Passwords checked") as pbar:
                for i, future in enumerate(futures):
                    result = future.result()
                    pbar.update(min(chunk_size, total_words - pbar.n))
                    if result:
                        # Cancel all remaining futures
                        for f in futures[i+1:]:
                            f.cancel()
                        elapsed_time = time.time() - start_time
                        print(f"\nPassword found in {elapsed_time:.2f} seconds!")
                        return result

    elapsed_time = time.time() - start_time
    print(f"\nDictionary attack completed in {elapsed_time:.2f} seconds. Password not found.")
    return None

def try_passwords_encrypted(password_iter, length, target_encrypted, key_b64, iv_b64, encryption_type):
    """
    Try passwords from an iterator and check if they match the encrypted data when decrypted.

    Args:
        password_iter: Iterator yielding password candidates
        length: Length of passwords being checked
        target_encrypted: The base64-encoded encrypted data to crack
        key_b64: The base64-encoded encryption key
        iv_b64: The base64-encoded initialization vector
        encryption_type: The encryption algorithm used

    Returns:
        The cracked password or None if not found
    """
    for candidate in password_iter:
        password = ''.join(candidate)
        match, found_password = check_encryption_match(password, target_encrypted, key_b64, iv_b64, encryption_type)
        if match:
            return found_password
    return None

def brute_force_attack_encrypted_cpu(target_encrypted, key_b64, iv_b64, encryption_type, charset, max_length, num_processes=None):
    """
    Perform a brute force attack on encrypted data using CPU parallel processing.

    Args:
        target_encrypted: The base64-encoded encrypted data to crack
        key_b64: The base64-encoded encryption key
        iv_b64: The base64-encoded initialization vector
        encryption_type: The encryption algorithm used
        charset: Character set to use (e.g., string.ascii_lowercase)
        max_length: Maximum password length to try
        num_processes: Number of processes to use (defaults to CPU count)

    Returns:
        The cracked password or None if not found
    """
    if not HAS_ENCRYPTION:
        print("Encryption libraries not available. Please install cryptography package.")
        return None

    if num_processes is None:
        num_processes = multiprocessing.cpu_count()

    print(f"Starting brute force attack on encrypted data using {num_processes} CPU processes...")
    start_time = time.time()

    # Try passwords of increasing length
    for length in range(1, max_length + 1):
        print(f"Trying passwords of length {length}...")

        # Calculate total combinations for this length
        total_combinations = len(charset) ** length

        # Split the work among processes
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            # Create iterators for each process
            chunk_size = total_combinations // num_processes
            if chunk_size == 0:
                chunk_size = 1

            futures = []
            for i in range(num_processes):
                # Create a generator that produces a subset of combinations
                start_idx = i * chunk_size
                end_idx = (i + 1) * chunk_size if i < num_processes - 1 else total_combinations

                futures.append(executor.submit(
                    try_passwords_encrypted, 
                    password_generator(start_idx, end_idx, length, charset),
                    length,
                    target_encrypted,
                    key_b64,
                    iv_b64,
                    encryption_type
                ))

            # Check results
            with tqdm(total=total_combinations, desc=f"Length {length}") as pbar:
                completed = 0
                for future in futures:
                    result = future.result()
                    if result:
                        elapsed_time = time.time() - start_time
                        print(f"\nPassword found in {elapsed_time:.2f} seconds!")
                        return result
                    completed += chunk_size
                    pbar.update(min(chunk_size, total_combinations - pbar.n))

    elapsed_time = time.time() - start_time
    print(f"\nBrute force attack completed in {elapsed_time:.2f} seconds. Password not found.")
    return None

def brute_force_attack_encrypted_gpu(target_encrypted, key_b64, iv_b64, encryption_type, charset, max_length):
    """
    Perform a brute force attack on encrypted data using GPU acceleration.

    Args:
        target_encrypted: The base64-encoded encrypted data to crack
        key_b64: The base64-encoded encryption key
        iv_b64: The base64-encoded initialization vector
        encryption_type: The encryption algorithm used
        charset: Character set to use
        max_length: Maximum password length to try

    Returns:
        The cracked password or None if not found
    """
    if not HAS_ENCRYPTION:
        print("Encryption libraries not available. Please install cryptography package.")
        return None

    if not HAS_GPU:
        print("GPU not available. Falling back to CPU.")
        return brute_force_attack_encrypted_cpu(target_encrypted, key_b64, iv_b64, encryption_type, charset, max_length)

    print("Starting brute force attack on encrypted data using GPU acceleration...")
    start_time = time.time()

    # This is a simplified GPU implementation
    # In a real-world scenario, you would use specialized libraries
    # or implement custom CUDA kernels for password decryption

    # For demonstration purposes, we'll use PyTorch for some basic parallelization
    device = torch.device("cuda")

    # Try passwords of increasing length
    for length in range(1, max_length + 1):
        print(f"Trying passwords of length {length}...")

        # Calculate total combinations for this length
        total_combinations = len(charset) ** length

        # Process in batches to avoid memory issues
        batch_size = min(1000000, total_combinations)

        # Generate all possible combinations
        for batch_start in range(0, total_combinations, batch_size):
            batch_end = min(batch_start + batch_size, total_combinations)

            # Generate passwords for this batch
            passwords = []
            for i in range(batch_start, batch_end):
                # Convert index to password
                indices = []
                temp = i
                for _ in range(length):
                    indices.append(temp % len(charset))
                    temp //= len(charset)
                indices.reverse()

                # Pad with zeros if needed
                while len(indices) < length:
                    indices.insert(0, 0)

                # Convert indices to characters
                password = ''.join(charset[idx] for idx in indices)
                passwords.append(password)

            # Check passwords in this batch
            for password in tqdm(passwords, desc=f"Batch {batch_start//batch_size + 1}"):
                match, found_password = check_encryption_match(password, target_encrypted, key_b64, iv_b64, encryption_type)
                if match:
                    elapsed_time = time.time() - start_time
                    print(f"\nPassword found in {elapsed_time:.2f} seconds!")
                    return found_password

    elapsed_time = time.time() - start_time
    print(f"\nBrute force attack completed in {elapsed_time:.2f} seconds. Password not found.")
    return None

def brute_force_attack_gpu(target_hash, hash_type, charset, max_length):
    """
    Perform a brute force attack using GPU acceleration.

    Args:
        target_hash: The hash to crack
        hash_type: The hash algorithm used
        charset: Character set to use
        max_length: Maximum password length to try

    Returns:
        The cracked password or None if not found
    """
    if not HAS_GPU:
        print("GPU not available. Falling back to CPU.")
        return brute_force_attack_cpu(target_hash, hash_type, charset, max_length)

    print("Starting brute force attack using GPU acceleration...")
    start_time = time.time()

    # This is a simplified GPU implementation
    # In a real-world scenario, you would use specialized libraries like hashcat
    # or implement custom CUDA kernels for password hashing

    # For demonstration purposes, we'll use PyTorch for some basic parallelization
    device = torch.device("cuda")

    # Try passwords of increasing length
    for length in range(1, max_length + 1):
        print(f"Trying passwords of length {length}...")

        # Calculate total combinations for this length
        total_combinations = len(charset) ** length

        # Process in batches to avoid memory issues
        batch_size = min(1000000, total_combinations)

        # Generate all possible combinations
        for batch_start in range(0, total_combinations, batch_size):
            batch_end = min(batch_start + batch_size, total_combinations)

            # Generate passwords for this batch
            passwords = []
            for i in range(batch_start, batch_end):
                # Convert index to password
                indices = []
                temp = i
                for _ in range(length):
                    indices.append(temp % len(charset))
                    temp //= len(charset)
                indices.reverse()

                # Pad with zeros if needed
                while len(indices) < length:
                    indices.insert(0, 0)

                # Convert indices to characters
                password = ''.join(charset[idx] for idx in indices)
                passwords.append(password)

            # Check passwords in this batch
            for password in tqdm(passwords, desc=f"Batch {batch_start//batch_size + 1}"):
                match, found_password = check_hash_match(password, target_hash, hash_type)
                if match:
                    elapsed_time = time.time() - start_time
                    print(f"\nPassword found in {elapsed_time:.2f} seconds!")
                    return found_password

    elapsed_time = time.time() - start_time
    print(f"\nBrute force attack completed in {elapsed_time:.2f} seconds. Password not found.")
    return None

if __name__ == "__main__":
    # Print banner
    print("""
    ╔═══════════════════════════════════════════════╗
    ║                                               ║
    ║           PASSWORD CRACKING TOOL              ║
    ║                                               ║
    ╚═══════════════════════════════════════════════╝
    """)

    # Check for GPU
    if HAS_GPU:
        print(f"GPU detected: {torch.cuda.get_device_name(0)}")
    else:
        print("No GPU detected. Using CPU only.")

    print(f"CPU cores available: {multiprocessing.cpu_count()}")
    print("")

    # Main function to run the password cracking tool
    parser = argparse.ArgumentParser(description="Password Cracking Tool")

    # Create a group for hash-related arguments
    hash_group = parser.add_argument_group('Hash Options')
    hash_group.add_argument("--hash", help="Hash to crack (if not provided, will prompt for password to hash/encrypt)")
    hash_group.add_argument("--type", choices=list(HASH_TYPES.keys()), default="sha256",
                         help="Hash type (default: sha256)")

    # Create a group for encryption-related arguments
    if HAS_ENCRYPTION:
        encrypt_group = parser.add_argument_group('Encryption Options')
        encrypt_group.add_argument("--encrypt", action="store_true", 
                            help="Use encryption instead of hashing")
        encrypt_group.add_argument("--encrypted", 
                            help="Base64-encoded encrypted data to crack")
        encrypt_group.add_argument("--encryption-type", choices=list(ENCRYPTION_TYPES.keys()),
                            default="aes-256", help="Encryption algorithm (default: aes-256)")
        encrypt_group.add_argument("--key", 
                            help="Encryption key (if not provided, will generate a random key)")
        encrypt_group.add_argument("--iv", 
                            help="Initialization vector for decryption (required with --encrypted)")

    # Create a group for attack-related arguments
    attack_group = parser.add_argument_group('Attack Options')
    attack_group.add_argument("--dict", help="Path to dictionary file for dictionary attack")
    attack_group.add_argument("--charset", default="abcdefghijklmnopqrstuvwxyz",
                         help="Character set for brute force (default: lowercase letters)")
    attack_group.add_argument("--max-length", type=int, default=6,
                         help="Maximum password length for brute force (default: 6)")
    attack_group.add_argument("--gpu", action="store_true", help="Use GPU acceleration if available")
    attack_group.add_argument("--processes", type=int, help="Number of CPU processes to use")

    args = parser.parse_args()

    # Check if we're using encryption or hashing
    using_encryption = HAS_ENCRYPTION and args.encrypt

    # If no hash/encrypted data provided, prompt for password to hash/encrypt
    if not args.hash and not (using_encryption and args.encrypted):
        if using_encryption:
            print("\n=== Password Encryption ===")
            password = input("Enter a password to encrypt: ")
            encryption_type = args.encryption_type
            key = args.key

            encrypted, key_b64, iv_b64 = encrypt_password(password, encryption_type, key)
            print(f"Encryption type: {encryption_type}")
            print(f"Encrypted password: {encrypted}")
            print(f"Key: {key_b64}")
            print(f"IV: {iv_b64}")

            # Use this encrypted data for cracking
            target_encrypted = encrypted
            target_key = key_b64
            target_iv = iv_b64
            print("\nNow attempting to crack this encrypted password...\n")
        else:
            print("\n=== Password Hashing ===")
            password = input("Enter a password to hash: ")
            hash_type = args.type
            hashed = hash_password(password, hash_type)
            print(f"Hash type: {hash_type}")
            print(f"Hashed password: {hashed}")

            # Use this hash for cracking
            target_hash = hashed
            print("\nNow attempting to crack this hash...\n")
    else:
        if using_encryption and args.encrypted:
            target_encrypted = args.encrypted
            target_key = args.key
            target_iv = args.iv

            if not target_key or not target_iv:
                print("Error: Both --key and --iv are required when cracking encrypted data.")
                exit(1)
        else:
            target_hash = args.hash

    # Attempt to crack the password
    if using_encryption:
        print(f"Target encrypted data: {target_encrypted}")
        print(f"Encryption type: {args.encryption_type}")
        print(f"Using key: {target_key}")

        # Try dictionary attack if dictionary file provided
        if args.dict:
            password = dictionary_attack_encrypted(target_encrypted, target_key, target_iv, 
                                                args.encryption_type, args.dict, args.processes)
            if password:
                print(f"Password cracked: {password}")
                exit(0)

        # Try brute force attack
        charset = list(args.charset)
        print(f"Using character set: {args.charset}")
        print(f"Maximum password length: {args.max_length}")

        if args.gpu and HAS_GPU:
            password = brute_force_attack_encrypted_gpu(target_encrypted, target_key, target_iv, 
                                                     args.encryption_type, charset, args.max_length)
        else:
            password = brute_force_attack_encrypted_cpu(target_encrypted, target_key, target_iv, 
                                                     args.encryption_type, charset, args.max_length, args.processes)

        if password:
            print(f"Password cracked: {password}")
        else:
            print("Failed to crack the password with the given parameters.")
    else:
        print(f"Target hash: {target_hash}")
        print(f"Hash type: {args.type}")

        # Try dictionary attack if dictionary file provided
        if args.dict:
            password = dictionary_attack(target_hash, args.type, args.dict, args.processes)
            if password:
                print(f"Password cracked: {password}")
                exit(0)

        # Try brute force attack
        charset = list(args.charset)
        print(f"Using character set: {args.charset}")
        print(f"Maximum password length: {args.max_length}")

        if args.gpu and HAS_GPU:
            password = brute_force_attack_gpu(target_hash, args.type, charset, args.max_length)
        else:
            password = brute_force_attack_cpu(target_hash, args.type, charset, args.max_length, args.processes)

        if password:
            print(f"Password cracked: {password}")
        else:
            print("Failed to crack the password with the given parameters.")

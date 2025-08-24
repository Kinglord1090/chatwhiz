"""
Encryption module for ChatWhiz providing AES encryption for chat files.
Ensures privacy and security of sensitive chat data.
"""

import os
import base64
import hashlib
import getpass
from typing import Optional, Union
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC


class ChatEncryptor:
    """
    AES encryption/decryption for chat files using Fernet (symmetric encryption).
    """
    
    def __init__(self, password: Optional[str] = None):
        """
        Initialize encryptor with password.
        
        Args:
            password: Encryption password. If None, will prompt user.
        """
        self.password = password
        self._fernet = None
    
    def _get_password(self) -> str:
        """Get password from user or environment."""
        if self.password:
            return self.password
        
        # Try environment variable first
        env_password = os.getenv('ENCRYPTION_PASSWORD')
        if env_password:
            return env_password
        
        # Prompt user
        return getpass.getpass("Enter encryption password: ")
    
    def _derive_key(self, password: str, salt: bytes) -> bytes:
        """
        Derive encryption key from password using PBKDF2.
        
        Args:
            password: User password
            salt: Random salt bytes
            
        Returns:
            Derived key bytes
        """
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,  # Recommended minimum
        )
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        return key
    
    def _get_fernet(self, salt: bytes) -> Fernet:
        """Get Fernet instance with derived key."""
        if self._fernet is None:
            password = self._get_password()
            key = self._derive_key(password, salt)
            self._fernet = Fernet(key)
        return self._fernet
    
    def encrypt_file(self, input_path: str, output_path: Optional[str] = None) -> str:
        """
        Encrypt a file.
        
        Args:
            input_path: Path to file to encrypt
            output_path: Output path (defaults to input_path + '.enc')
            
        Returns:
            Path to encrypted file
        """
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        if output_path is None:
            output_path = input_path + '.enc'
        
        # Generate random salt
        salt = os.urandom(16)
        
        # Read input file
        with open(input_path, 'rb') as f:
            data = f.read()
        
        # Encrypt data
        fernet = self._get_fernet(salt)
        encrypted_data = fernet.encrypt(data)
        
        # Write encrypted file (salt + encrypted data)
        with open(output_path, 'wb') as f:
            f.write(salt)  # First 16 bytes are salt
            f.write(encrypted_data)
        
        print(f"File encrypted: {input_path} -> {output_path}")
        return output_path
    
    def decrypt_file(self, input_path: str, output_path: Optional[str] = None) -> str:
        """
        Decrypt a file.
        
        Args:
            input_path: Path to encrypted file
            output_path: Output path (defaults to input_path without '.enc')
            
        Returns:
            Path to decrypted file
        """
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        if output_path is None:
            if input_path.endswith('.enc'):
                output_path = input_path[:-4]  # Remove '.enc'
            else:
                output_path = input_path + '.dec'
        
        # Read encrypted file
        with open(input_path, 'rb') as f:
            salt = f.read(16)  # First 16 bytes are salt
            encrypted_data = f.read()
        
        if len(salt) != 16:
            raise ValueError("Invalid encrypted file format")
        
        # Decrypt data
        fernet = self._get_fernet(salt)
        try:
            decrypted_data = fernet.decrypt(encrypted_data)
        except Exception as e:
            raise ValueError(f"Decryption failed. Wrong password? Error: {e}")
        
        # Write decrypted file
        with open(output_path, 'wb') as f:
            f.write(decrypted_data)
        
        print(f"File decrypted: {input_path} -> {output_path}")
        return output_path
    
    def encrypt_text(self, text: str) -> str:
        """
        Encrypt text and return base64 encoded result.
        
        Args:
            text: Text to encrypt
            
        Returns:
            Base64 encoded encrypted text with salt
        """
        # Generate random salt
        salt = os.urandom(16)
        
        # Encrypt text
        fernet = self._get_fernet(salt)
        encrypted_data = fernet.encrypt(text.encode('utf-8'))
        
        # Combine salt and encrypted data, then base64 encode
        combined = salt + encrypted_data
        return base64.b64encode(combined).decode('ascii')
    
    def decrypt_text(self, encrypted_text: str) -> str:
        """
        Decrypt base64 encoded encrypted text.
        
        Args:
            encrypted_text: Base64 encoded encrypted text
            
        Returns:
            Decrypted text
        """
        try:
            # Decode base64
            combined = base64.b64decode(encrypted_text.encode('ascii'))
            
            # Extract salt and encrypted data
            salt = combined[:16]
            encrypted_data = combined[16:]
            
            # Decrypt
            fernet = self._get_fernet(salt)
            decrypted_data = fernet.decrypt(encrypted_data)
            
            return decrypted_data.decode('utf-8')
        
        except Exception as e:
            raise ValueError(f"Decryption failed: {e}")
    
    def is_encrypted_file(self, file_path: str) -> bool:
        """
        Check if a file appears to be encrypted by this system.

        Args:
            file_path: Path to check

        Returns:
            True if file appears encrypted
        """
        if not os.path.exists(file_path):
            return False

        # Check file extension first - if it's .enc, likely encrypted
        if file_path.endswith('.enc'):
            return True

        # For common text formats, check if they're readable as text
        if file_path.endswith(('.txt', '.json', '.csv', '.yaml', '.yml')):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    # Try to read first 100 characters
                    sample = f.read(100)
                    # If we can read it as text and it contains printable characters, it's not encrypted
                    if sample and all(ord(c) < 128 and (c.isprintable() or c.isspace()) for c in sample):
                        return False
            except (UnicodeDecodeError, Exception):
                # If we can't read it as text, it might be encrypted
                pass

        # For binary files or files that failed text reading, use more sophisticated check
        try:
            with open(file_path, 'rb') as f:
                # Check if file is large enough to contain salt
                if os.path.getsize(file_path) < 16:
                    return False

                # Read first 16 bytes (potential salt)
                salt = f.read(16)

                # Check if it looks like our encryption format
                # Our encrypted files start with 16 random bytes (salt)
                # followed by Fernet encrypted data
                if len(set(salt)) >= 12:  # Salt should be very random
                    # Read more data to check if it looks like Fernet format
                    f.seek(16)
                    data_sample = f.read(50)
                    if data_sample and len(set(data_sample)) >= 10:  # Encrypted data should be random
                        return True

                return False

        except Exception:
            return False
    
    def generate_key_file(self, key_path: str, password: Optional[str] = None) -> str:
        """
        Generate a key file for easier key management.
        
        Args:
            key_path: Path to save key file
            password: Password to derive key from
            
        Returns:
            Path to key file
        """
        if password is None:
            password = self._get_password()
        
        # Generate salt and derive key
        salt = os.urandom(16)
        key = self._derive_key(password, salt)
        
        # Save salt and key hash for verification
        key_data = {
            'salt': base64.b64encode(salt).decode('ascii'),
            'key_hash': hashlib.sha256(key).hexdigest()
        }
        
        import json
        with open(key_path, 'w') as f:
            json.dump(key_data, f, indent=2)
        
        print(f"Key file generated: {key_path}")
        return key_path
    
    def verify_password(self, key_path: str, password: Optional[str] = None) -> bool:
        """
        Verify password against a key file.
        
        Args:
            key_path: Path to key file
            password: Password to verify
            
        Returns:
            True if password is correct
        """
        if not os.path.exists(key_path):
            raise FileNotFoundError(f"Key file not found: {key_path}")
        
        if password is None:
            password = self._get_password()
        
        import json
        with open(key_path, 'r') as f:
            key_data = json.load(f)
        
        # Derive key and check hash
        salt = base64.b64decode(key_data['salt'].encode('ascii'))
        key = self._derive_key(password, salt)
        key_hash = hashlib.sha256(key).hexdigest()
        
        return key_hash == key_data['key_hash']


def encrypt_chat_file(
    input_path: str,
    password: Optional[str] = None,
    output_path: Optional[str] = None
) -> str:
    """
    Convenience function to encrypt a chat file.
    
    Args:
        input_path: Path to chat file
        password: Encryption password
        output_path: Output path
        
    Returns:
        Path to encrypted file
    """
    encryptor = ChatEncryptor(password)
    return encryptor.encrypt_file(input_path, output_path)


def decrypt_chat_file(
    input_path: str,
    password: Optional[str] = None,
    output_path: Optional[str] = None
) -> str:
    """
    Convenience function to decrypt a chat file.
    
    Args:
        input_path: Path to encrypted file
        password: Decryption password
        output_path: Output path
        
    Returns:
        Path to decrypted file
    """
    encryptor = ChatEncryptor(password)
    return encryptor.decrypt_file(input_path, output_path)


def is_file_encrypted(file_path: str) -> bool:
    """
    Check if a file is encrypted.
    
    Args:
        file_path: Path to check
        
    Returns:
        True if file appears encrypted
    """
    encryptor = ChatEncryptor()
    return encryptor.is_encrypted_file(file_path)

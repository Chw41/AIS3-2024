import random
from Crypto.Util.number import getPrime
from secret import flag
def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def generate_keypair(keysize):
    p = getPrime(keysize)
    q = getPrime(keysize)
    n = p * q
    phi = (p-1) * (q-1)
    
    e = random.randrange(1, phi)
    g = gcd(e, phi)
    while g != 1:
        e = random.randrange(1, phi)
        g = gcd(e, phi)
    d = pow(e, -1, phi)
    return ((e, n), (d, n))

def encrypt(pk, plaintext):
    key, n = pk
    cipher = [pow(ord(char), key, n) for char in plaintext]
    return cipher

def decrypt(pk, ciphertext):
    key, n = pk
    plain = [chr(pow(char, key, n)) for char in ciphertext]
    return ''.join(plain)

public, private = generate_keypair(512)
encrypted_msg = encrypt(public, flag)
decrypted_msg = decrypt(private, encrypted_msg)

print("Public Key:", public)
print("Encrypted:", encrypted_msg)
# print("Decrypted:", decrypted_msg)
# Dynamic sized chunks hash
In the part, The Rabin-Karp rolling hash algorithm is excellent at finding a pattern in a large file or stream, but there is an even more interesting use case: creating content based chunks of a file to detect the changed blocks without doing a full byte-by-byte comparison.
To calculate the next window’s hash all we need to do is to remove the window’s first element's ‘hash value’ from the current hash and add the new to it.
Even though a simple mathematical addition would do fine, it’s usually not preferred because it can cause high collision rate among the hash values. However, Rabin-Karp algorithm typically works with the powers of a prime number to do calculate the hash. So we use this method to implement hashing with a prime number. One of the usages of the Rabin-Karp algorithm is pattern matching: finding a short string sample in a large text corpus.


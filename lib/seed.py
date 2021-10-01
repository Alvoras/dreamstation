import whirlpool


def make_seed_from_file(filepath):
    with open(filepath, "rb") as f:
        content = f.read()

    return make_seed_from_str(content)


def make_seed_from_str(str_seed):
    # We use whirlpool to make up for the high collision rate of hashCode, caused by
    # similarities in consecutive characters
    h = whirlpool.new(str_seed)
    return hashcode64(h.hexdigest())


# Python implementation of Java's hashCode function
# Adapted to return an int64
def hashcode64(s):
    h = 0
    for c in s:
        h = (31 * h + ord(c)) & 0xFFFFFFFFFFFFFFFF
    return ((h + 0x8000000000000000) & 0xFFFFFFFFFFFFFFFF) - 0x8000000000000000

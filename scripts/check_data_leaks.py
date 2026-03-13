with open("./meta/train_curated.txt") as f1, open("./meta/test_curated.txt") as f2:
    common = set(f1.read().splitlines()) & set(f2.read().splitlines())

for line in common:
    print(line)

print(bool(common))
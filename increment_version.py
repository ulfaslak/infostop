import re, datetime

# Read metadata
with open("infostop/metadata.py", "r") as fp:
    metadata = fp.read()

# Update version
version = re.findall(r'__version__ = "(.*)"', metadata)[0].split(".")
version[-1] = str(int(version[-1]) + 1)
version = ".".join(version)
metadata = re.sub(r'__version__ = "(.*)"', f'__version__ = "{version}"', metadata)

# Update year
year = str(datetime.datetime.now().year)
metadata = re.sub(r"(?<=Copyright )(\d{4})", year, metadata)

# Write metadata
with open("infostop/metadata.py", "w") as fp:
    fp.write(metadata)

print(f"Updated version to {version}")

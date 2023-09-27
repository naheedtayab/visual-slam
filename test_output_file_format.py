import re
import sys

output_file = open(sys.argv[1], 'r')
lines = output_file.readlines()
format_followed = True
for line in lines:
    line_matched = re.match(r"^[0-9]+:(\s[0-9]+.[0-9]{2})*$\n", line)
    format_followed = bool(line_matched)
    if format_followed == False:
        break;

if format_followed:
    print("File conforms with required format.")
else:
    print("Please check the file format.")
output_file.close()

import re
test_list = ["fvalues.pth","fvalues0.pth","aaaa.pth"]
pattern = r"fvalues.*\.pth"

for file_name in test_list:
    print(re.match(pattern,file_name))
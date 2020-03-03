import re

text = 'The phone number of the agent is 321-123-7891. Call Soon to this phone!!!'

#print('321-123-7891' in text)

pattern = 'phone'

""" print(re.search(pattern, text)) """

matches = re.finditer(pattern, text)
""" for i in matches:
    print(i) """

#pattern = r'\d\d\d-\d\d\d-\d\d\d\d'
pattern = r'(\d{3})-(\d{3})-(\d{4})'

print(text)
print(re.search(pattern, text).group(1))

print(re.findall(r'\d$', 'The cat 2 is in the car asdat'))

phrase = 'There are 3 numbers in this 587 phares 12312'

print(phrase)

list = []
list.append(2)
list.append(3)
list.append(4)
print(list)

if str(type(list) == "<class 'list'>"):
    print('hello')

print(str(type(4)))

new_list = [1/i for i in list]

print(list)

print(min(list))
print(max(list))
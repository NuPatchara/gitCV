list1 = {1:[[1,2,3,4],[100,100,100],[200,200,200]]}
# values = set(map(lambda x:x[1], list1))
newlist = [[y[1] for y in list1 if y[1]==x] for x in list1]

print newlist
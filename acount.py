i = 1
acount = 0

f = open('count_re.txt', 'r')
for line in f.readlines():
    line_strip = line.rstrip('\n')
    acount = acount + float(line_strip)
    if i == 320:
        print(acount)
        i = 1
        acount = 0
    i +=1
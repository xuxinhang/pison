from __init__ import LogoLexer, LogoParser


testcases = {}
# some cases are disabled due to progressing implement

testcases['example1.txt'] = '''\
fd 60 rt 120 fd 60 rt 120 fd 60 rt 120
'''

testcases['example2.txt'] = '''\
cs pu setxy -60 60 pd home rt 45 fd 85 lt 135 fd 120
'''

testcases['example3.txt'] = '''\
make "first_programmer "Ada_Lovelace
print :first_programmer
'''

testcases['example4.txt'] = '''\
make "angle 0
    repeat 1000 [fd 3 rt :angle make "angle :angle + 7]
'''

testcases['expression.txt'] = '''\
make "size 81/9
print 2*3
print :size - 4
'''

testcases['flower.txt'] = '''\
repeat 8 [rt 45 repeat 6 [repeat 90 [fd 1 rt 2] rt 90]] ht
'''

testcases['fractal.txt'] = '''\
to triangle :size
    to fractal :size
        repeat 3 [fd :size rt 120]
        if :size < 3 [stop] ; the procedure stops if size is too small
    end
    repeat 3 [fd :size fractal :size/2 fd :size rt 120]
end
'''

# testcases['logo_fail_1.txt'] = '''\
# print "if
# print "print
# print "repeat
# '''

# testcases['logo_fail_2.txt'] = '''\
# print "1234
# '''

# testcases['logo_feature_butfail.txt'] = '''\
# ;In logo, characters in [] after print command should be output as a string.
# print[hello world]              ;should display "hello world"
# print[if 1>0 [print[1>0]]]      ;should display "if 1>0 [print[1>0]]'''

# testcases['logo_feature_butfail_2.txt'] = '''\
# ;In logo, characters in [] after print command should be output as a string,include blank space and unicode and number
# print[hello world]              ;should display "hello world"
# print[hello ▒▒]                 ;should display "hello ▒▒"
# '''

testcases['make.txt'] = '''\
make "size 60
'''

testcases['octagon.txt'] = '''\
repeat 8 [rt 45 repeat 6 [repeat 90 [fd 1 rt 2] rt 90]] ht
'''

testcases['procedure1.txt'] = '''\
to random_walk
    repeat 100 [fd random 80 rt 90]
end
'''

testcases['procedure2.txt'] = '''\
to square :size
    repeat 4 [fd :size rt 90] ; where is the turtle when this step completes?
end

to floor :size
    repeat 2 [fd :size rt 90 fd :size * 2 rt 90]
end

to house
    floor 60 fd 60 floor 60 ; where is the turtle at this point?
    pu fd 20 rt 90 fd 20 lt 90 pd
    square 20
    pu rt 90 fd 60 lt 90 pd
    square 20
end
'''

testcases['random.txt'] = '''\
repeat 100 [fd random 80 rt 90]
    repeat 1000 [fd 4 rt random 360]
'''

testcases['recursive1.txt'] = '''\
to star to walk_the_stars
    repeat 5 [fd 10 rt 144] fd 20 rt random 360
end

star
walk_the_stars
end
'''

testcases['repeat.txt'] = '''\
repeat 3 [fd 60 rt 120]
'''

testcases['spiral.txt'] = '''\
to spiral :size :angle
    if :size > 100 [stop]
    forward :size
    right :angle
    spiral :size + 2 :angle
end
'''

testcases['spiral2.txt'] = '''\
for [i 10 100 10] [fd :i rt 90] ht
'''

testcases['tree.txt'] = '''\
to left_side
    rt 20 fd 20 lt 20 fd 60
end

to top_side
    rt 90 fd 25 rt 90
end

to right_side
    fd 60 lt 20 fd 20 rt 20
end

to return_start
    rt 90 fd 40
    rt 90
end

to trunk
    left_side
    top_side
    right_side
    return_start
end

to center_top
    pu
    fd 80
    rt 90
    fd 20
    lt 90
    pd
end

to circle
    repeat 360 [fd 1 rt 1]
end

to tree
    pu bk 100 pd
    trunk
    center_top
    left 90
    circle
end
'''

testcases['tree2.txt'] = '''\
TO tree :size
   if :size < 5 [forward :size back :size stop]
   forward :size/3
   left 30 tree :size*2/3 right 30
   forward :size/6
   right 25 tree :size/2 left 25
   forward :size/3
   right 25 tree :size/2 left 25
   forward :size/6
   back :size
END
clearscreen
tree 150
'''

lexer = LogoLexer()
parser = LogoParser(debug=True)

for tn, tc in testcases.items():
    print('')
    print(':: ' + tn + ' ::')
    lexer.input(tc)
    # print(list(lexer))
    ast_root = parser.parse(lexer)
    print(ast_root)


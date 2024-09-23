
import sys
import math


def get_parameter_vectors():
    '''
    This function parses e.txt and s.txt to get the  26-dimensional multinomial
    parameter vector (characters probabilities of English and Spanish) as
    descibed in section 1.2 of the writeup

    Returns: tuple of vectors e and s
    '''
    #Implementing vectors e,s as lists (arrays) of length 26
    #with p[0] being the probability of 'A' and so on
    e=[0]*26
    s=[0]*26

    with open('e.txt',encoding='utf-8') as f:
        for line in f:
            #strip: removes the newline character
            #split: split the string on space character
            char,prob=line.strip().split(" ")
            #ord('E') gives the ASCII (integer) value of character 'E'
            #we then subtract it from 'A' to give array index
            #This way 'A' gets index 0 and 'Z' gets index 25.
            e[ord(char)-ord('A')]=float(prob)
    f.close()

    with open('s.txt',encoding='utf-8') as f:
        for line in f:
            char,prob=line.strip().split(" ")
            s[ord(char)-ord('A')]=float(prob)
    f.close()

    return (e,s)

def shred(filename):
    #Using a dictionary here. You may change this to any data structure of
    #your choice such as lists (X=[]) etc. for the assignment
    X= {chr(ord('A') + i): 0 for i in range(26)}
    with open (filename,encoding='utf-8') as f:
        # TODO: add your code here
        #read file content
        content = f.read()
        
        #all chars -> uppercase for case-folding
        content = content.upper()
        
        #for every character
        for char in content:
            #if char A-Z
            if 'A' <= char <= 'Z':
                X[char] += 1
    return X

#Q1
print("Q1")
letterCounts = shred('letter.txt')
for char, count in letterCounts.items():
    print(f"{char} {count}")

#Q2
print("Q2")
e, s = get_parameter_vectors()
X1log_e1 = letterCounts['A'] * math.log(e[0]) if letterCounts['A'] > 0 else 0
X1log_s1 = letterCounts['A'] * math.log(s[0]) if letterCounts['A'] > 0 else 0
print(f"{X1log_e1:.4f}")
print(f"{X1log_s1:.4f}")

#Q3
print("Q3")
pYEnglish = math.log(0.6)
PYSpanish = math.log(0.4)
Fenglish = pYEnglish + sum(X1log_e * count for X1log_e, count in zip(map(math.log, e), letterCounts.values()))
Fspanish = PYSpanish + sum(X1log_s * count for X1log_s, count in zip(map(math.log, s), letterCounts.values()))
print(f"{Fenglish:.4f}")
print(f"{Fspanish:.4f}")

#Q4
print("Q4")
if Fspanish - Fenglish >= 100:
    PYEnglishwithX = 0
elif Fspanish - Fenglish <= -100:
    PYEnglishwithX = 1
else:
    ratio = 1 / (1 + math.exp(Fspanish - Fenglish))
    PYEnglishwithX = 1 / (1 + math.exp(Fspanish - Fenglish))

print(f"{PYEnglishwithX:.4f}")

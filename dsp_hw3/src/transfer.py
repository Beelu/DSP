# coding: Big5-HKSCS
import sys

with open(sys.argv[1], 'r', encoding='big5-hkscs') as f:
    lines = f.readlines()

zhuyin_list = ['ㄅ', 'ㄆ', 'ㄇ', 'ㄈ', 'ㄉ', 'ㄊ', 'ㄋ', 'ㄌ', 'ㄍ', 'ㄎ', 'ㄏ', 'ㄐ', 'ㄑ', 'ㄒ', 'ㄓ', 'ㄔ', 'ㄕ', 'ㄖ', 'ㄗ', 'ㄘ', 'ㄙ', 'ㄧ', 'ㄨ', 'ㄩ', 'ㄚ', 'ㄛ', 'ㄜ', 'ㄝ', 'ㄞ', 'ㄟ', 'ㄠ', 'ㄡ', 'ㄢ', 'ㄣ', 'ㄤ', 'ㄥ', 'ㄦ']
inverse_lines1 = [zhuyin_list[i] + "\t" for i in range(len(zhuyin_list))]        # 這個存37個注音的inverse
inverse_lines2 = [line[0] + "\t" + line[0] for line in lines]        # 這個存big5 to big5每個字
inverse_str = []
for line in lines:
    zhuyin_for_word = line[2:].split('/')           # 注音的list(不包含國字)
    for i in range(len(zhuyin_list)):               # 對於每個注音
        for z in zhuyin_for_word:                   # 比對每個破音念法的字首 
            if z[0]==zhuyin_list[i]:
                inverse_lines1[i] += (line[0] + " ")
                break


# 移除沒有對應字的注音
for i in range(len(inverse_lines1)):
    if i==len(inverse_lines1):
        break
    if len(inverse_lines1[i])==2:
        inverse_lines1.pop(i)

final_inverse_line = inverse_lines1 + inverse_lines2

for i in range(len(final_inverse_line)):
    final_inverse_line[i] += '\n'

with open(sys.argv[2], 'w', encoding='big5-hkscs') as f2:
    f2.writelines(final_inverse_line)
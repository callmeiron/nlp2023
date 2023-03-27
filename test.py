import os

stopword_file=open('cn_punctuation.txt',  "r",  encoding='utf-8')
stopwordlist =  stopword_file.read().split('\n')
stopword_file.close()


corpus =  []
datapath = "./jyxstxtqj_downcc.com"
filelist = os.listdir(datapath)

for filename in filelist:
    filepath = datapath +'/'+ filename
    with open(filepath, "r", encoding="gb18030") as file:
      filecontext  =  file.read()
      filecontext  =  filecontext.replace("本书来自www.cr173.com免费txt小说下载站\n更多更新免费电子书请关注www.cr173.com", '')
      filecontext  =  filecontext.replace("本书来自www.cr173.com免费txt小说下载站", '')
      corpus.append(filecontext)
      file.close()


fw  =  open('all_sentence.txt',  'w',  encoding='utf-8')

for filecontext in corpus:
       for  stopword  in  stopwordlist:
           filecontext = filecontext.replace(stopword,'')
       filecontext = filecontext.replace('\n', '')
       filecontext = filecontext.replace(' ', '')
       fw.write(filecontext.strip(' ')  + '\n')

fw.close()





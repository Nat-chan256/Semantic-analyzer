from nltk import word_tokenize
from nltk import sent_tokenize
from pymorphy2 import MorphAnalyzer
import sys
import string
import numpy as np

FILE_NAME = "graphTask.txt"
OUTPUT_FILE_NAME = FILE_NAME.split(".")[0] + " matrix.txt"

numerals = {"первый": "1", 
            "второй": "2", 
            "третий": "3", 
            "четвёртый": "4", 
            "пятый": "5", 
            "шестой": "6", 
            "седьмой": "7", 
            "восьмой": "8", 
            "девятый": "9", 
            "десятый": "10",
            "один": "1",
            "два": "2",
            "три": "3",
            "четыре": "4",
            "пять": "5",
            "шесть": "6",
            "семь": "7",
            "восемь": "8",
            "девять": "9",
            "десять": "10"}

fromQuantityToOrdinal = { "один": "первый",
        "два": "второй",
        "три": "третий",
        "четыре": "четвертый",
        "пять": "пятый",
        "шесть": "шестой",
        "семь": "седьмой",
        "восемь": "восьмой",
        "девять": "девятый",
        "десять": "десятый"}

#Сбор конструкций на основе числительных и соответствующих им существительных
def createConstruction(sent, numrIndex, nounIndex, nounCase = "None"): #sent - список слов
    morph = MorphAnalyzer()
    pNoun = morph.parse(sent[nounIndex])[0]
    pNumr = morph.parse(sent[numrIndex])[0]
    construction = [sent[numrIndex]]

    #Устанавливаем падеж существительного в зависимости от наличия предлога "с" перед числительным
    if numrIndex > 0 and (sent[numrIndex - 1] == "с" or sent[numrIndex - 1] == "между") and nounCase == "None":
        nounCase = "ablt"
    if nounCase == "None":
        nounCase = pNumr.tag.case
    construction.append(pNoun.inflect({pNumr.tag.gender, nounCase, pNumr.tag.number}).word)

    #Собираем остальные компоненты конструкции
    for i in range(numrIndex, len(sent)):
        if i == numrIndex or i == nounIndex:
            continue
        if sent[i] in string.punctuation or sent[i] == "и":
            break
        if isSubject(sent[i]):
            subjParse = morph.parse(sent[i])[0]
            construction.append(subjParse.inflect({pNoun.tag.case, pNoun.tag.gender, pNoun.tag.number}).word)
        else:
            construction.append(sent[i])

    return construction

def isNumr(word):
    morph = MorphAnalyzer()
    p = morph.parse(word)[0]
    return p.tag.POS == "ADJF" and p.normal_form in numerals.keys() or p.tag.POS == "NUMR"

def findLastNoun(sent, indexToStart): #sent - список слов
    morph = MorphAnalyzer()
    for i in range(indexToStart, -1, -1):
        p = morph.parse(sent[i])[0]
        if p.tag.POS == "NOUN":
            return sent[i]
    return ""
   
#Из списка слов делает список предложений, где каждое предложение - список слов
def groupIntoSentences(wordsList):
    resultList = []
    curSent = []
    for word in wordsList:
        curSent.append(word)
        if word == ".":
            resultList.append(curSent)
            curSent = []
    return resultList

#Функция, проверяющая, совпадают ли слова из разборов в роде, числе и падеже
def matchInGNC(parse1, parse2): 
    return parse1.tag.gender == parse2.tag.gender and parse1.tag.number == parse2.tag.number and parse1.tag.case == parse2.tag.case

def isSubject(word):
    morph = MorphAnalyzer()
    p = morph.parse(word)[0]
    return p.tag.POS == "VERB" or p.tag.POS == "PRTS" or p.tag.POS == "PRTF"

#Возвращает фрагмент предложения в виде списка слов
def extractCurrentFragment(fullSent, insideFragmentIndex): #fullSent - список слов
    fragment = []
    i = insideFragmentIndex
    while fullSent[i] != "," and i >= 0: #Собираем первую часть фрагмента до позиции insideFragmentIndex
        fragment.insert(0,fullSent[i])
        i -= 1
    i = insideFragmentIndex + 1
    while i < len(fullSent) and fullSent[i] != "." and fullSent[i] != ",":
        fragment.append(fullSent[i])
        i += 1
    return fragment

#Возвращает существительное, соответствующее числительному, имеющему индекс numrIndex
def findCorrespondingNoun(sent, numrIndex): #sent - предложение, представленное в виде списка слов
    morph = MorphAnalyzer()
    #Рассматриваем случай, когда существительное опущено из-за того, что данное числительное является однородным членом
    if numrIndex+1 != len(sent) and (sent[numrIndex+1] == "и" or sent[numrIndex+1] == ","):  
        #Ищем существительное после числительного
        for i in range(numrIndex + 1, len(sent)): 
            curWordParse = morph.parse(sent[i])[0]
            if curWordParse.tag.POS == "NOUN":
                return sent[i]
    #Ищем существительное перед числительным
    for i in range(numrIndex-1, -1, -1):
        curWordParse = morph.parse(sent[i])[0]
        if curWordParse.tag.POS == "NOUN":
            return sent[i]

#Возвращает фрагменты предложения sent, идущие до и после фрагмента separatingFragment
def divideSentence(sent, separatingFragment): #sent и separatingFragment - списки слов
    beforePart = []
    afterPart = []
    middlePart = []
    fragmentIndex = 0

    for i in range(len(sent)):
        if fragmentIndex < len(separatingFragment) and sent[i] != separatingFragment[fragmentIndex]: #Слово находится до фрагмента
            beforePart.extend(middlePart)
            beforePart.append(sent[i])
            fragmentIndex = 0
        elif fragmentIndex == len(separatingFragment): #Слово находится после фрагмента
            afterPart.append(sent[i])
        else: #Слово находится внутри фрагмента
            middlePart.append(sent[i])
            fragmentIndex += 1

    return beforePart, afterPart

#Стандартизирует предложение, включающее конструкцию формата "...вершина под номером..."
def proccessSentWithWordNumber(sent): #sent - список слов
    #Находим фрагмент предложения, в котором находится слово "номер"
    morph = MorphAnalyzer()
    fragment = []
    numberWordIndex = int()
    for i in range(len(sent)):
        if morph.parse(sent[i])[0].normal_form == "номер":
            fragment = extractCurrentFragment(sent, i)
            numberWordIndex = i
            break

    #Находим в данном фрагменте числительные
    numrs = []
    for i in range(len(fragment)):
        if isNumr(fragment[i]):
            numrParse = morph.parse(fromQuantityToOrdinal[fragment[i]])[0]
            numrs.append(numrParse.inflect({"femn"}).word)

    #Строим конструкции вида "Первая вершина", "Вторая вершина" и т.д. из имеющихся числительных
    constructions = [] #Список конструкций, где каждая конструкция - это список строк
    vertexParse = morph.parse("вершина")[0]
    for numr in numrs:
        curConstr = [numr]
        if numberWordIndex > 0 and (sent[numberWordIndex-1] == "с" or sent[numberWordIndex-1] == "между"): #Проверяем, не нужно ли поставить существительное творительный падеж
            curConstr.append(vertexParse.inflect({"ablt"}).word)
        else:
            curConstr.append(vertexParse.inflect({morph.parse(numr)[0].tag.case}).word) #Ставим существительное в форму, соответствующую
                                                                                                                         #форме числительного
        constructions.append(curConstr)

    #Записываем часть фрагмента, предшествующую "вершине с номером"
    resultFrag = []
    i = int()
    for i in range(len(fragment)):
        if morph.parse(fragment[i])[0].normal_form == "вершина":
            break;
        resultFrag.append(fragment[i])
    
    #Заменяем числительные соответствующими конструкциями
    for j in range(i, len(fragment)):
        if morph.parse(fragment[j])[0].normal_form == "вершина" or fragment[j] == "с" \
        or morph.parse(fragment[j])[0].normal_form == "номер":
            continue

        if not isNumr(fragment[j]):
            resultFrag.append(fragment[j])
            continue

        resultFrag.extend(constructions[0])
        constructions.remove(constructions[0])

    beforeProccessedFrag, afterProccessedFrag = divideSentence(sent, fragment)

    return beforeProccessedFrag + resultFrag + afterProccessedFrag

def insertNounsAfterNumrs(sents): #sents - список предложений, где каждое предложение - список слов
    resultText = [] #Результирующий текст представим в виде списка предложений, где каждое предложение - это список слов
    morph = MorphAnalyzer()
    for sent in sents:
        currentSentence = sent.copy()
        indexToCont = -1 #Вспомогательный индекс для пропуска уже обработанных слов
        for i in range(len(sent)):
            if i >= len(currentSentence):
                break
            if i < indexToCont:
                continue

             #Обрабатываем случай, когда в предложении есть конструкции формата "...вершина под номером..."
            if includesWord(currentSentence, "номер"):
                currentSentence = proccessSentWithWordNumber(currentSentence)
                resultText.extend(currentSentence)
                break


            p = morph.parse(currentSentence[i])[0] #Ищем числительные
            if not isNumr(currentSentence[i]): #Если текущее слово - не числительное
                resultText.append(currentSentence[i]) #добавляем его в результирующий список без изменений
                continue

            #Если нашли-таки числительное, то ищем соответствующее ему существительное

            #Обрабатываем случай, когда числительное обозначает длину дуги
            if currentSentence[i-1] == "-" or currentSentence[i-1] == "равна":
                resultText.append(currentSentence[i])
                continue

            nextWordParse = morph.parse(currentSentence[i])[0]
            if i+1 != len(currentSentence):
                nextWordParse = morph.parse(currentSentence[i+1])[0]
            indexToCont = i #Индекс, с которого продолжим обработку

            #Обрабатываем случай с однородными числительными
            nextNextWordParse = morph.parse(currentSentence[i])[0]
            if (i+2 < len(currentSentence)):
                nextNextWordParse = morph.parse(currentSentence[i+2])[0]
            if i+1 != len(currentSentence) and ((currentSentence[i+1] == "и" or currentSentence[i+1] == ",") and i+2 != len(currentSentence) and isNumr(currentSentence[i+2]) \
                and matchInGNC(p, nextNextWordParse)):
                constructs = [] #Составляем список конструкций формата [["Первая", "вершина"], ["Вторая", вершина"] и т.д.]
                curConstr = []
                j = int() #Индекс существительного
                k = int() #Идекс ближайшего знака препинания
                nounCase = "None" #Падеж однородных членов
                noun = findCorrespondingNoun(currentSentence, i)
                nounParse = morph.parse(noun)[0]
          
                curConstr.append(currentSentence[i])
                if i > 0 and (currentSentence[i-1] == "с" or currentSentence[i-1] == "между"): #Проверяем, не нужно ли поставить существительное творительный падеж
                    curConstr.append(nounParse.inflect({"ablt", p.tag.gender, p.tag.number}).word)
                    nounCase = "ablt"
                else:
                    curConstr.append(nounParse.inflect({p.tag.case, p.tag.gender, p.tag.number}).word) #Ставим существительное в форму, соответствующую форме числительного

                #Проверяем, нужно ли вставлять в текущую конструкцию что-то, кроме существительного
                curFrag = extractCurrentFragment(currentSentence, i)
                if not includesWord(curFrag, "между"):
                    for k in range(j+1, len(currentSentence)): #Собираем все слова до ближайшего знака препинания или предлога "и"
                        if currentSentence[k] in string.punctuation or currentSentence[k] == "и":
                            break
                        if isSubject(currentSentence[k]):
                            subjParse = morph.parse(currentSentence[k])[0]
                            curConstr.append(subjParse.inflect({p.tag.case, p.tag.gender, p.tag.number}).word)
                        else:
                            curConstr.append(currentSentence[k])
                constructs.append(curConstr)
                
                for l in range(i+2, j): #Собираем остальные конструкции
                    parse = morph.parse(currentSentence[l])[0]
                    if (isNumr(currentSentence[l])):
                        constructs.append(createConstruction(currentSentence, l, j, nounCase))
                for c in range(len(constructs)):
                    resultText.extend(constructs[c])
                    if c != len(constructs) - 1:
                        resultText.append("и")
                indexToCont = k

            #Обрабатываем случай, когда существительное опущено
            elif i+1 != len(currentSentence) and nextWordParse.tag.POS != "NOUN":
                noun = findLastNoun(currentSentence, i-1) 
                resultText.append(currentSentence[i])
                if noun != "":
                    nounParse = morph.parse(noun)[0]
                    if i > 0 and (currentSentence[i-1] == "с" or currentSentence[i-1] == "между"): #Проверяем, не нужно ли поставить существительное творительный падеж
                        resultText.append(nounParse.inflect({p.tag.number, "ablt"}).word)
                    else:
                        resultText.append(nounParse.inflect({p.tag.number, p.tag.case}).word)
            else:
                resultText.append(currentSentence[i])


    return groupIntoSentences(resultText) 

def replaceNumrsWithNumbers(text): #text = [["Первое", "предложение", "."], ["Второе", "предложение", "."], и т. д.]
    resultText = []
    morph = MorphAnalyzer()
    for sent in text:
        curSent = []
        for word in sent:
            p = morph.parse(word)[0]
            if (p.normal_form in numerals.keys()):
                curSent.append(numerals[p.normal_form])
            else:
                curSent.append(word)
        resultText.append(curSent)
        curSent = []
    return resultText

#Считает количество слов, которые могут быть сказуемыми
def predicateCount(wordsList): 
    result = 0
    morph = MorphAnalyzer()
    for word in wordsList:
        if isSubject(word):
            result += 1
    return result

#Проверка, разделяет ли запятая части сложного предложения
def isSentsSeparator(sent, commaIndex): #sent - список слов
    part1 = sent[:commaIndex]
    part2 = sent[commaIndex+1:]
    return predicateCount(part1) >= 1 and predicateCount(part2) >= 1


def findLastNounPhrase(sent): #sent - список слов
    morph = MorphAnalyzer()
    result = []
    for i in range(len(sent)-1, -1, -1):
        p = morph.parse(sent[i])[0]
        if p.tag.POS == "NOUN": #Когда нашли существительное
            result = [sent[i]]
            for j in range(i-1, -1, -1): #ищем стоящие перед ним прилагательные или числительные
                prevWordParse = morph.parse(sent[j])[0]
                if prevWordParse.tag.POS == "ADJF" or sent[j].isnumeric():
                    result.insert(0, sent[j])
                else:
                    break
            break
    return result

#Деление сложного предложения на несколько простых
def divideDifficultSentence(sent): #sent - список слов
    resultList = []
    morph = MorphAnalyzer()
    for i in range(len(sent)):
        if sent[i] == "," or sent[i] == "и":
            if isSentsSeparator(sent, i): #Проверяем, разделяет ли данная запятая простые предложения
                sent1 = sent[:i]
                sent1.append(".")
                sent2 = sent[i+1:]
                #sent2.append(".")
                p = morph.parse(sent2[0])[0]

                if (p.normal_form == "который"): #Если второе предлжение начинается со слова "который"
                    subject = findLastNounPhrase(sent1) #то заменяем его на соответствующую именную группу
                    sent2.pop(0)
                    for j in range(len(subject)-1, -1, -1):
                        npParse = morph.parse(subject[j])[0]
                        wordToInsert = subject[j]
                        if not subject[j].isnumeric():
                            wordToInsert = npParse.inflect({p.tag.case, p.tag.number}).word
                        sent2.insert(0, wordToInsert)

                sent1 = divideDifficultSentence(sent1) #Снова проверяем полученные предложения
                sent2 = divideDifficultSentence(sent2)

                resultList.extend(sent1)
                resultList.extend(sent2)
                return resultList

        if sent[i] == ".": #Обрабатываем случа, когда было передано простое предложение
            return [sent]

def divideIntoSimpleSentences(text): #text = [["Первое", "предложение", "."], ["Второе", "предложение", "."], и т. д.]
    resultText = []
    for sent in text:
        if predicateCount(sent) <= 1: #Если в предложении не больше одного сказуемого
            resultText.append(sent) #не обрабатываем его
            continue
        simpleSents = divideDifficultSentence(sent)
        resultText.extend(simpleSents)
    return resultText

def standardizeText(text): #text - переменная типа string
    sents = sent_tokenize(text)
    textBySents = [[word for word in word_tokenize(sent)] for sent in sents]

    tempRes = insertNounsAfterNumrs(textBySents)
    t1 = insertNounsAfterNumrs(tempRes)

    t2 = replaceNumrsWithNumbers(t1)

    t3 = divideIntoSimpleSentences(t2)

    return t3

#Функция, проверяющая наличие в предложении слова, начальная форма которого совпадает с word
def includesWord(sent, word): #sent - список слов
    morph = MorphAnalyzer()
    for w in sent:
        p = morph.parse(w)[0]
        if p.normal_form == word:
            return True
    return False

#Находит первое вхождение числа, возвращает его индекс
def findNumber(sent): #sent - список слов
    for i in range(len(sent)):
        if sent[i].isnumeric():
            return i
    return -1

#Находит последнее вхождение числа, возвращает его индекс
def findLastNum(sent): #sent - список слов
    for i in range(len(sent)-1, -1, -1):
        if sent[i].isnumeric():
            return i
    return -1

#Функция, извлекающая данные из текста
def extractData(standardizedText): #standardizedText - стандартизированный текст в виде списка предложений, где предложение - это список слов
    result = {}
    for sent in standardizedText:
        numInd = findNumber(sent)
        sliceStartIndex = numInd+1

        while numInd != -1:
            #Проверяем, что означает полученное число
            if numInd < len(sent) - 1:
                if (sent[numInd+1].lower() == "вершин" or sent[numInd+1].lower() == "вершины") and includesWord(sent, "граф"): #Количество вершин в графе
                    result["dim"] = int(sent[numInd])
                elif sent[numInd+1].lower() == "вершина" and (includesWord(sent, "соединить") or includesWord(sent, "связать")) and findNumber(sent[sliceStartIndex:]) != -1: #Наличие ребра
                    otherVertNum = findNumber(sent[sliceStartIndex:])
                    if otherVertNum != -1:
                        otherVertNum += sliceStartIndex
                        sliceStartIndex = otherVertNum+1
                        v1 = min(int(sent[numInd]), int(sent[otherVertNum]))
                        v2 = max(int(sent[numInd]), int(sent[otherVertNum]))
                        result[f"{v1}-{v2}"] = 1
                elif numInd > 1 and sent[numInd-1].lower() == "и" and sent[numInd+1].lower() == "вершиной": #Наличие ребра, обработка однородных членов
                    prevNumInd = findLastNum(sent[:numInd-1])
                    while not (includesWord(sent[prevNumInd:numInd], "соединить") or includesWord(sent[prevNumInd:numInd], "связать")):
                        prevNumInd = findLastNum(sent[:prevNumInd-1])
                        if prevNumInd == -1:
                            break
                    if prevNumInd != -1:
                        v1 = min(int(sent[prevNumInd]), int(sent[numInd]))
                        v2 = max(int(sent[prevNumInd]), int(sent[numInd]))
                        result[f"{v1}-{v2}"] = 1
                elif sent[numInd+1] in string.punctuation and includesWord(sent, "длина"): #Длина ребра
                    firstVertInd = findLastNum(sent[:numInd-1])
                    secondVertInd = findLastNum(sent[:firstVertInd-1])
                    if firstVertInd != -1 and secondVertInd != -1:
                        v1 = min(int(sent[firstVertInd]), int(sent[secondVertInd]))
                        v2 = max(int(sent[firstVertInd]), int(sent[secondVertInd]))
                        result[f"{v1}-{v2}"] = int(sent[numInd])


            numInd = findNumber(sent[sliceStartIndex:])
            if numInd != -1:
                numInd += sliceStartIndex
            sliceStartIndex = numInd+1


    return result


def buildMatrix(dict):
    if "dim" in dict.keys():
        matrix = np.zeros((dict["dim"], dict["dim"])) #Создаем нулевую матрицу
        for key in dict.keys(): #Заполняем матрицу 
            if key == "dim":
                continue
            verteces = key.split("-")
            v1 = int(verteces[0])-1
            v2 = int(verteces[1])-1
            matrix[v1][v2] = dict[key]
            matrix[v2][v1] = dict[key]
        return matrix
    return [[]]

#Читаем задание из файла
text = str()
try:
    file = open(FILE_NAME, "r", encoding="UTF-8")
    text = file.read()
    file.close()

except FileNotFoundError:
    print(f"Файл {FILE_NAME} не найден")
    sys.exit()


#Стандартизируем текст
st = standardizeText(text)
#Извлекаем данные
dict = extractData(st)
#Строим матрицу
matrix = buildMatrix(dict)

#Запись матрицы в файл
outputFile = open(OUTPUT_FILE_NAME, "w", encoding="UTF-8")
for i in range(len(matrix)):
    for j in range(len(matrix[i])):
        outputFile.write(str(int(matrix[i][j])) + " ")
    outputFile.write("\n")
outputFile.close()

print(f"Файл {OUTPUT_FILE_NAME} успешно создан.")

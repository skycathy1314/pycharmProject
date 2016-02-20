import nltk
content = open("training_data.txt", "r").read()
res_write = open("output.txt", "w+")
c_lists = content.split("\n")
lst = []
for index in range(len(c_lists)):
    c_list = c_lists[index].split(",")
    lst.append(tuple(c_list))

all_words = set(word for message in lst for word in nltk.word_tokenize(message[0]))
get_features = [({word: (word in nltk.word_tokenize(x[0])) for word in all_words}, x[1]) for x in lst]
classifier = nltk.NaiveBayesClassifier.train(get_features)
test = open("test_data.txt", "r").read()
test_lists = test.split("\n")
test_lst = []
test_ref = []
for index in range(len(test_lists)):
    test_list = test_lists[index].split(",")
    test_lst.append(test_list[0])
    test_ref.append(test_list[1])

def test_classify(sentence):
    features = {word: (word in nltk.word_tokenize(sentence)) for word in all_words}
    return classifier.classify(features)

test_result = []
for t in test_lst:
    test_result.append(test_classify(t))

def get_accuracy(res,ref):
    count = 0.0
    for i in range(len(res)):
        if(res[i] == ref[i]):
            count+=1
    return count/len(res)
acc = get_accuracy(test_result,test_ref)

print "classier's accuracy: ",
print acc
res_write.write("classier's accuracy: ")
res_write.write(str(acc)+"\n")
examples = open("example.txt", "r").read()
ex_lists = examples.split("\n")
for i in range(len(ex_lists)):
    t_classify = test_classify(ex_lists[i])
    print "classication result " + str(i+1) + ":",
    print t_classify
    res_write.write("classication result " + str(i+1) + ":")
    res_write.write(t_classify+"\n")
res_write.close()

en_id2word = {}
en_word2id = {}
en_word2cnt = {}
zh_id2word = {}
zh_word2id = {}
zh_word2cnt = {}
en_id = 1
zh_id = 1
with open("corpus.txt", 'r') as fr:
	for line in fr:
		[zh_sen, en_sen] = line.strip().split(" ||| ")
		for w in zh_sen.split(' '):
			if w not in zh_word2id:
				zh_word2id[w] = str(zh_id)
				zh_id2word[str(zh_id)] = w
				zh_word2cnt[w] = 1
				zh_id += 1
			else:
				zh_word2cnt[w] += 1
		for w in en_sen.split(' '):
			if w not in en_word2id:
				en_word2id[w] = str(en_id)
				en_id2word[str(en_id)] = w
				en_word2cnt[w] = 1
				en_id += 1
			else:
				en_word2cnt[w] += 1

with open("TS.snt", 'w') as fw_1:
	with open("ST.snt", 'w') as fw_2:
		with open("corpus.txt", 'r') as fr:
			for line in fr:
				[zh_sen, en_sen] = line.strip().split(" ||| ")
				array = []
				item = []
				for w in zh_sen.split(' '):
					array.append(zh_word2id[w])
				item.append(' '.join(array) + "\n")
				array = []
				for w in en_sen.split(' '):
					array.append(en_word2id[w])
				item.append(' '.join(array) + "\n")
				fw_1.write("1\n")
				fw_1.write(item[1])
				fw_1.write(item[0])
				fw_2.write("1\n")
				fw_2.write(item[0])
				fw_2.write(item[1])

with open("S.vcb", 'w') as fw:
	for i in range(1, zh_id):
		fw.write(str(i) + " " + zh_id2word[str(i)] + " " + str(zh_word2cnt[zh_id2word[str(i)]]) + "\n")

with open("T.vcb", 'w') as fw:
	for i in range(1, en_id):
		fw.write(str(i) + " " + en_id2word[str(i)] + " " + str(en_word2cnt[en_id2word[str(i)]]) + "\n")

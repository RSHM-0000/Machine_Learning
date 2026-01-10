# Q4

word=input("Enter the word: ")
freq={} #We use dictionary to get values

for ch in word: #For every letter in the word
  freq[ch]=freq.get(ch,0)+1 # Each ch is key, frequency detects and inserts as values against its key

max_freq=max(freq,key=freq.get) #We get the key with max frequency
max_value=freq[max_freq]#To store the max frequency

print("The letter with highest frequency is: ",max_freq)
print("The frequency of its appearance is: ",max_value)

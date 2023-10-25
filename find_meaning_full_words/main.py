import nltk
from nltk.corpus import words

nltk.download('punkt')
nltk.download('words')

# Initialize NLTK and download the words corpus
nltk.download('words')

# Define an input string
input_string = "This is a sample sentence with meaningful words. rtre tert choco im in the cave LLOOFJF"

# Tokenize the input string into words
tokens = nltk.word_tokenize(input_string)

# Get a list of English words from the NLTK corpus
english_words = set(words.words())

# Filter the tokens to keep only meaningful words
meaningful_words = [word for word in tokens if word.lower() in english_words]

# Display the meaningful words
print("Meaningful words found in the input string:")
print(meaningful_words)

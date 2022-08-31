"""ex02_03.py"""
import random
from datetime import datetime

def creat_words():
    return "linux python robot system network computer service programming".split()

def word_jumble(words):
    random.seed(datetime.now())
    word = random.choice(words) 
    jumble = random.sample(word, len(word))
    jumble = ''.join(jumble)
    return word, jumble

def main():
    words = creat_words()
    word, jumble = word_jumble(words)
    print(f"The jumble word is: [{jumble:>15}]")

    guess = input(f"Write your guess: ")
    if guess.lower() == word:
        print(f"Corret! The {jumble} is [{guess:^15}]")
    else:
        print(f"Incorrect! The {jumble} is [{word:<15}]")

if __name__ == '__main__':
    main()
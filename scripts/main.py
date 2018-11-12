import word_embedding_training as wet
import label_image as li
import argparse

if __name__ == '__main__':

    image = "example.jpg"
    text_filename = "files/text"
    result_filename = "files/results"
    training = True
    num_words = 5

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", help="image to be processed")
    parser.add_argument("--text", help="text used for embedding")
    parser.add_argument("--result", help="file where to store results")
    parser.add_argument("--training", help="is training needed? (Y or N)")
    parser.add_argument("--num_words", help="number of near words to search")
    args = parser.parse_args()

    if args.image:
        image = args.image
    if args.text:
        text_filename = args.text
    if args.result:
        result_filename = args.result
    if args.training:
        training = (False, True)[args.training is 'Y']
    if args.num_words:
        num_words = args.num_words

    _, _, top_k, _, labels = li.get_labels(image)

    best_label = labels[top_k[0]]

    print("The best label given by the image classifier to " + image + " is " + best_label)

    (vectors, words) = wet.word_embedding(text_filename, result_filename, training)
    results = wet.get_near_words(best_label, num_words, vectors, words)
    if len(results) == 1:
        print("\nThe nearest word to " + labels[top_k[0]] + " is:")
    else:
        print("\nThe " + str(len(results)) + " nearest words to " + labels[top_k[0]] + " are:")
    for i in range(len(results)):
        print(results[i])

    wet.plot(vectors, words, with_nearest_words=True, target_word=best_label, nearest_words=results)
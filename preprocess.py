import json
import datetime
import statistics 
import pickle
from functools import reduce

data = eval(open('final_updated', encoding='utf8').read())
pos = set(open('lexicon/positive.txt').read().split('\n'))
neg = set(open('lexicon/negative.txt').read().split('\n'))
category_dict = {'Chiptune': 1, 'R&B': 2, 'Ready-to-wear': 3, 'Literary Spaces': 4, 'Animals': 5, 'Indie Rock': 6, 'Cookbooks': 7, 'Video Games': 8, 'Publishing': 9, 'Spaces': 10, 'Civic Design': 11, 'Stationery': 12, 'Nature': 13, 'Web': 14, 'Translations': 15, 'Public Art': 16, 'Webcomics': 17, 'Crochet': 18, 'Mobile Games': 19, 'Shorts': 20, 'DIY Electronics': 21, 'Literary Journals': 22, 'Glass': 23, 'Classical Music': 24, 'Festivals': 25, 'Letterpress': 26, 'Horror': 27, 'Crafts': 28, 'Product Design': 29, 'Typography': 30, 'Comedy': 31, 'Embroidery': 32, "Farmer's Markets": 33, 'Pop': 34, 'Events': 35, 'Residencies': 36, 'Animation': 37, 'Interactive Design': 38, 'Painting': 39, 'Textiles': 40, 'Narrative Film': 41, 'Woodworking': 42, 'Fabrication Tools': 43, 'Installations': 44, 'Movie Theaters': 45, 'Sculpture': 46, 'Graphic Design': 47, 'Television': 48, 'Wearables': 49, 'Mixed Media': 50, 'Sound': 51, 'Faith': 52, 'Camera Equipment': 53, 'Software': 54, 'Fine Art': 55, 'Immersive': 56, 'Video Art': 57, 'Commissions': 58, 'Anthologies': 59, 'Robots': 60, 'Community Gardens': 61, 'Photobooks': 62, 'Architecture': 63, 'Pottery': 64, 'Latin': 65, 'Conceptual Art': 66, 'Fiction': 67, 'Music Videos': 68, 'Gadgets': 69, 'Thrillers': 70, 'People': 71, 'Science Fiction': 72, 'Theater': 73, 'Childrenswear': 74, 'Design': 75, 'Playing Cards': 76, 'Print': 77, 'Jazz': 78, 'Webseries': 79, 'Fashion': 80, 'Journalism': 81, 'Young Adult': 82, 'Restaurants': 83, 'Printing': 84, 'Food Trucks': 85, 'Hip-Hop': 86, 'Comic Books': 87, 'Country & Folk': 88, 'Apparel': 89, 'Gaming Hardware': 90, 'Graphic Novels': 91, 'Drama': 92, 'Punk': 93, 'Tabletop Games': 94, 'Metal': 95, 'Bacon': 96, 'Food': 97, 'Radio & Podcasts': 98, 'Zines': 99, 'Kids': 100, 'Video': 101, 'Couture': 102, 'Photo': 103, 'Documentary': 104, 'Romance': 105, 'Performances': 106, 'World Music': 107, 'Electronic Music': 108, 'Pet Fashion': 109, 'Action': 110, 'Weaving': 111, 'Hardware': 112, 'Plays': 113, 'Nonfiction': 114, 'Ceramics': 115, 'Farms': 116, 'Vegan': 117, 'Live Games': 118, 'Film & Video': 119, 'Small Batch': 120, 'Places': 121, 'Candles': 122, 'Audio': 123, 'Jewelry': 124, 'Art Books': 125, 'Periodicals': 126, 'Puzzles': 127, 'Photography': 128, 'Family': 129, 'Rock': 130, 'Comics': 131, 'Footwear': 132, "Children's Books": 133, 'Makerspaces': 134, 'Blues': 135, 'Workshops': 136, 'Music': 137, 'Performance Art': 138, 'Space Exploration': 139, 'Dance': 140, 'Games': 141, 'DIY': 142, 'Flight': 143, 'Drinks': 144, 'Calendars': 145, 'Academic': 146, 'Experimental': 147, 'Knitting': 148, 'Art': 149, '3D Printing': 150, 'Make 100': 151, 'Poetry': 152, 'Accessories': 153, 'Technology': 154, 'Digital Art': 155, 'Illustration': 156, 'Musical': 157, 'Fantasy': 158, 'Apps': 159}

features = []

for item in data.values():
    if 'description' not in item.keys():
        continue
    if 'pledge' in item.keys():
        if(len(item['pledge']) > 0):
            pledge_num = len(item['pledge'])
            pledge_max = max(item['pledge'])
            pledge_min = min(item['pledge'])
            pledge_mean = statistics.mean(item['pledge'])
        else:
             pledge_num = pledge_max = pledge_mean = pledge_min = 0
    else:
        pledge_num = pledge_max = pledge_mean = pledge_min = 0
    duration = item['period']
    goal = item['goal']
    category = category_dict[item['category']] if 'category' in item.keys() else 0
    back_experience = item['backNum']
    project_experience = item['createNum']
    backed_failed_rate = item['backFail']
    created_failed_rate = item['createFail']
    comment_num = item['comment']
    collaborators_num = item['collaborators']
    websites_num = item['websites']
    fb_friends_num = item['fb'] if 'fb' in item.keys() else 0
    disclosure_fb = 1 if fb_friends_num > 0 else 0
    pictures_num = item['imgnum'] if 'imgnum' in item.keys() else 0
    videos_num = item['vidnum'] if 'vidnum' in item.keys() else 0
    faces_num = item['faces'] if 'faces' in item.keys() else 0
    topic_popularity = 0 if 'popular' not in item.keys() else item['popular']

    if 'text' in item.keys():
        text = item['text']
        gunning_fog_score = text['Gunning Fog Score'] if 'Gunning Fog Score' in text.keys() else 0
        sentences_num = text['No. of sentences'] if 'No. of sentences' in text.keys() else len(item['description'].split('.'))
        words_num = text['No. of words'] if 'No. of words' in text.keys() else len(item['description'].split())
        avg_words = text['Average words  per sentence'] if 'Average words  per sentence' in text.keys() else round(float(words_num)/float(sentences_num), 2)
        complex_words_num = text['No. of complex words'] if 'No. of complex words' in text.keys() else 0
        avg_syllables = text['Average syllables per word'] if 'Average syllables per word' in text.keys() else 0
    else:
        text = item['description']
        gunning_fog_score = 0
        sentences_num = len(text.split('.')) 
        words_num = len(text.split())
        avg_words = round(float(words_num)/float(sentences_num), 2)
        complex_words_num = 0
        avg_syllables = 0

    positive = round(
        reduce(lambda x, y: x + 1 if y.strip() in pos else x, item['description'].split(), 0) / (
                    words_num + 1), 2)
    negative = round(
        reduce(lambda x, y: x + 1 if y.strip() in neg else x, item['description'].split(), 0) / (
                    words_num + 1), 2)

    features.append([disclosure_fb, fb_friends_num, websites_num, back_experience, project_experience, created_failed_rate, 
        backed_failed_rate, comment_num, collaborators_num, pledge_num, pledge_min, pledge_max, pledge_mean, duration,
        goal, category, words_num, sentences_num, avg_words, complex_words_num, avg_syllables, positive, negative, 
        topic_popularity, gunning_fog_score, pictures_num, videos_num, faces_num])
    #print(key)

pickle.dump(features, open('features.p', 'wb'))


    


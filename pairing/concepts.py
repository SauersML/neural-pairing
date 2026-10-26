"""Concepts and sentence templates.

CONCEPTS: (name, phrase, category) triples. The name is how an instruction refers to a concept; the phrase fills the
slot {X} of each sentence in SENTENCES. TEST_CONCEPTS: indices of the held-out concepts, every fourth concept of each
category starting with the second. EXTRA: further training-only concepts (category 'extra'), none of them held out.
build_pools.py writes the concept pools of data/pools.json from CONCEPTS and TEST_CONCEPTS; steer_read.py draws its
concepts from CONCEPTS and EXTRA and fits their directions on SENTENCES.
"""

CONCEPTS = [
    # animals
    ('cat', 'cats', 'animal'), ('dog', 'dogs', 'animal'), ('horse', 'horses', 'animal'), ('rabbit', 'rabbits', 'animal'),
    ('elephant', 'elephants', 'animal'), ('lion', 'lions', 'animal'), ('tiger', 'tigers', 'animal'), ('bear', 'bears', 'animal'),
    ('wolf', 'wolves', 'animal'), ('fox', 'foxes', 'animal'), ('monkey', 'monkeys', 'animal'), ('snake', 'snakes', 'animal'),
    ('shark', 'sharks', 'animal'), ('whale', 'whales', 'animal'), ('dolphin', 'dolphins', 'animal'), ('eagle', 'eagles', 'animal'),
    ('owl', 'owls', 'animal'), ('penguin', 'penguins', 'animal'), ('frog', 'frogs', 'animal'), ('spider', 'spiders', 'animal'),
    ('bee', 'bees', 'animal'), ('butterfly', 'butterflies', 'animal'), ('cow', 'cows', 'animal'), ('pig', 'pigs', 'animal'),
    ('sheep', 'sheep', 'animal'), ('chicken', 'chickens', 'animal'), ('mouse', 'mice', 'animal'), ('deer', 'deer', 'animal'),
    # food and drink
    ('pizza', 'pizza', 'food'), ('bread', 'bread', 'food'), ('cheese', 'cheese', 'food'), ('chocolate', 'chocolate', 'food'),
    ('coffee', 'coffee', 'food'), ('tea', 'tea', 'food'), ('apple', 'apples', 'food'), ('banana', 'bananas', 'food'),
    ('strawberry', 'strawberries', 'food'), ('rice', 'rice', 'food'), ('pasta', 'pasta', 'food'), ('soup', 'soup', 'food'),
    ('cake', 'cake', 'food'), ('cookie', 'cookies', 'food'), ('sushi', 'sushi', 'food'), ('burger', 'burgers', 'food'),
    ('salad', 'salad', 'food'), ('honey', 'honey', 'food'), ('egg', 'eggs', 'food'), ('milk', 'milk', 'food'),
    ('wine', 'wine', 'food'), ('beer', 'beer', 'food'), ('tomato', 'tomatoes', 'food'), ('potato', 'potatoes', 'food'),
    # objects and vehicles
    ('car', 'cars', 'object'), ('bicycle', 'bicycles', 'object'), ('train', 'trains', 'object'), ('airplane', 'airplanes', 'object'),
    ('boat', 'boats', 'object'), ('chair', 'chairs', 'object'), ('lamp', 'lamps', 'object'), ('phone', 'phones', 'object'),
    ('computer', 'computers', 'object'), ('book', 'books', 'object'), ('pencil', 'pencils', 'object'), ('clock', 'clocks', 'object'),
    ('key', 'keys', 'object'), ('guitar', 'guitars', 'object'), ('piano', 'pianos', 'object'), ('camera', 'cameras', 'object'),
    ('umbrella', 'umbrellas', 'object'), ('hat', 'hats', 'object'), ('shoe', 'shoes', 'object'), ('bottle', 'bottles', 'object'),
    ('knife', 'knives', 'object'), ('mirror', 'mirrors', 'object'), ('candle', 'candles', 'object'), ('sword', 'swords', 'object'),
    # places
    ('France', 'France', 'place'), ('Japan', 'Japan', 'place'), ('Brazil', 'Brazil', 'place'), ('Egypt', 'Egypt', 'place'),
    ('India', 'India', 'place'), ('Canada', 'Canada', 'place'), ('Italy', 'Italy', 'place'), ('Russia', 'Russia', 'place'),
    ('China', 'China', 'place'), ('Mexico', 'Mexico', 'place'), ('Paris', 'Paris', 'place'), ('London', 'London', 'place'),
    ('Tokyo', 'Tokyo', 'place'), ('New York', 'New York', 'place'), ('Rome', 'Rome', 'place'), ('Africa', 'Africa', 'place'),
    ('the desert', 'the desert', 'place'), ('the forest', 'the forest', 'place'), ('mountains', 'mountains', 'place'),
    ('the ocean', 'the ocean', 'place'), ('rivers', 'rivers', 'place'), ('the beach', 'the beach', 'place'),
    ('islands', 'islands', 'place'), ('the moon', 'the moon', 'place'),
    # nature and weather
    ('rain', 'rain', 'nature'), ('snow', 'snow', 'nature'), ('storms', 'storms', 'nature'), ('thunder', 'thunder', 'nature'),
    ('the sun', 'the sun', 'nature'), ('stars', 'stars', 'nature'), ('fire', 'fire', 'nature'), ('ice', 'ice', 'nature'),
    ('wind', 'wind', 'nature'), ('clouds', 'clouds', 'nature'), ('rainbows', 'rainbows', 'nature'), ('volcanoes', 'volcanoes', 'nature'),
    ('earthquakes', 'earthquakes', 'nature'), ('flowers', 'flowers', 'nature'), ('trees', 'trees', 'nature'), ('gold', 'gold', 'nature'),
    # feelings
    ('love', 'love', 'feeling'), ('anger', 'anger', 'feeling'), ('fear', 'fear', 'feeling'), ('joy', 'joy', 'feeling'),
    ('sadness', 'sadness', 'feeling'), ('jealousy', 'jealousy', 'feeling'), ('pride', 'pride', 'feeling'), ('guilt', 'guilt', 'feeling'),
    ('hope', 'hope', 'feeling'), ('boredom', 'boredom', 'feeling'), ('loneliness', 'loneliness', 'feeling'), ('surprise', 'surprise', 'feeling'),
    ('calm', 'calm', 'feeling'), ('excitement', 'excitement', 'feeling'), ('shame', 'shame', 'feeling'), ('curiosity', 'curiosity', 'feeling'),
    # people
    ('doctors', 'doctors', 'person'), ('teachers', 'teachers', 'person'), ('soldiers', 'soldiers', 'person'), ('farmers', 'farmers', 'person'),
    ('pirates', 'pirates', 'person'), ('kings', 'kings', 'person'), ('queens', 'queens', 'person'), ('babies', 'babies', 'person'),
    ('wizards', 'wizards', 'person'), ('robots', 'robots', 'person'), ('astronauts', 'astronauts', 'person'), ('chefs', 'chefs', 'person'),
    ('police officers', 'police officers', 'person'), ('nurses', 'nurses', 'person'), ('artists', 'artists', 'person'),
    ('scientists', 'scientists', 'person'),
    # topics
    ('music', 'music', 'topic'), ('mathematics', 'mathematics', 'topic'), ('money', 'money', 'topic'), ('war', 'war', 'topic'),
    ('death', 'death', 'topic'), ('time', 'time', 'topic'), ('religion', 'religion', 'topic'), ('politics', 'politics', 'topic'),
    ('science', 'science', 'topic'), ('art', 'art', 'topic'), ('poetry', 'poetry', 'topic'), ('sports', 'sports', 'topic'),
    ('football', 'football', 'topic'), ('chess', 'chess', 'topic'), ('history', 'history', 'topic'), ('medicine', 'medicine', 'topic'),
    ('law', 'law', 'topic'), ('technology', 'technology', 'topic'), ('language', 'language', 'topic'), ('philosophy', 'philosophy', 'topic'),
    ('dance', 'dance', 'topic'), ('crime', 'crime', 'topic'), ('school', 'school', 'topic'), ('family', 'family', 'topic'),
    # colours
    ('red', 'the color red', 'color'), ('blue', 'the color blue', 'color'), ('green', 'the color green', 'color'),
    ('yellow', 'the color yellow', 'color'), ('purple', 'the color purple', 'color'), ('black', 'the color black', 'color'),
    ('white', 'the color white', 'color'), ('pink', 'the color pink', 'color'),
]


def _held_out():
    seen = {}
    out = []
    for i, (_, _, cat) in enumerate(CONCEPTS):
        j = seen.get(cat, 0)
        seen[cat] = j + 1
        if j % 4 == 1:
            out.append(i)
    return out


TEST_CONCEPTS = _held_out()

# Sentences with one slot {X} that read naturally for any concept phrase.
SENTENCES = [
    "I can't stop thinking about {X}.",
    "The whole lecture was about {X}.",
    "My favorite thing to talk about is {X}.",
    "She wrote a long poem about {X}.",
    "This picture reminds me of {X}.",
    "He has always been fascinated by {X}.",
    "Today's topic is {X}.",
    "The documentary focused on {X}.",
    "We spent the whole evening discussing {X}.",
    "There is a new book about {X}.",
    "Everyone in the room was talking about {X}.",
    "My dream last night was full of {X}.",
    "The painting on the wall showed {X}.",
    "Her notebook was full of drawings of {X}.",
    "The museum opened an exhibit on {X}.",
    "Nothing interests me more than {X}.",
    "The children asked a hundred questions about {X}.",
    "His essay on {X} won first prize.",
    "I searched the internet for {X}.",
    "The song is about {X}.",
    "Grandpa told us old stories about {X}.",
    "The podcast episode was all about {X}.",
    "They made a movie about {X}.",
    "My teacher knows everything about {X}.",
    "The magazine cover featured {X}.",
    "I read an article about {X} this morning.",
    "Our class project is on {X}.",
    "Somebody left a note about {X} on my desk.",
    "The comedian made jokes about {X}.",
    "Let me tell you about {X}.",
    "The professor gave a talk on {X}.",
    "Last summer I learned a lot about {X}.",
    "The poster in the hallway was about {X}.",
    "She can talk about {X} for hours.",
    "The quiz had ten questions about {X}.",
    "My little brother is obsessed with {X}.",
    "The library has a whole shelf on {X}.",
    "The tour guide kept mentioning {X}.",
    "In my diary I wrote about {X}.",
    "The game show category was {X}.",
    "We watched a video about {X}.",
    "Our neighbor collects pictures of {X}.",
    "The meeting turned into a debate about {X}.",
    "Mom bought me a puzzle with {X} on it.",
    "The headline was about {X}.",
    "A stranger on the bus talked to me about {X}.",
    "The final exam covered {X}.",
    "I drew a sketch of {X} in class.",
]

# EXTRA: a word of _EXTRA_PLURAL has its plural as its phrase (word + 's' unless _PLURAL_FIX gives it), a word of
# _EXTRA_MASS is its own phrase, and _EXTRA_PHRASED lists (name, phrase).
_EXTRA_PLURAL = """giraffe zebra kangaroo koala panda gorilla camel hippo rhino crocodile turtle octopus jellyfish crab lobster
ant mosquito bat squirrel hedgehog raccoon otter beaver seal parrot pigeon crow swan duck goose flamingo peacock hawk goat
donkey llama buffalo moose salmon goldfish snail worm scorpion dragon unicorn dinosaur carrot onion mushroom cucumber grape
cherry peach pear watermelon pineapple mango coconut lemon avocado peanut almond pancake waffle taco sausage sandwich donut
pie bed sofa door window ladder hammer needle rope bucket basket wallet ring necklace watch backpack suitcase pillow blanket
towel toothbrush comb spoon fork plate cup kettle oven fridge television radio microphone keyboard printer battery flashlight
compass map telescope microscope balloon kite drum violin trumpet flute harp helmet tent anchor crown coin flag bell statue
ticket envelope stamp truck bus motorcycle helicopter rocket submarine tractor ambulance taxi ship canoe skateboard sled
swamp cave lake waterfall valley canyon farm village castle church hospital airport library prison museum zoo garden kitchen
bathroom office factory market stadium bridge tower pyramid tornado hurricane flood drought wave rock crystal diamond leaf
rose tulip sunflower cactus seed sunset magnet lawyer engineer firefighter pilot sailor spy detective priest monk ninja
knight cowboy viking clown magician athlete singer actor writer poet painter dancer student grandparent twin neighbor
stranger ghost zombie vampire witch alien giant fairy angel demon president emperor princess thief prisoner refugee tourist
fisherman miner plumber carpenter dentist surgeon veterinarian judge birthday wedding funeral holiday election protest
revolution puzzle""".split()
_PLURAL_FIX = {'toothbrush': 'toothbrushes', 'battery': 'batteries', 'spy': 'spies', 'leaf': 'leaves',
               'thief': 'thieves', 'fisherman': 'fishermen',
               'cactus': 'cacti', 'octopus': 'octopuses', 'jellyfish': 'jellyfish', 'moose': 'moose', 'salmon': 'salmon',
               'goldfish': 'goldfish', 'buffalo': 'buffaloes', 'mosquito': 'mosquitoes',
               'tornado': 'tornadoes', 'church': 'churches', 'watch': 'watches', 'bus': 'buses',
               'peach': 'peaches', 'sandwich': 'sandwiches', 'witch': 'witches', 'princess': 'princesses',
               'compass': 'compasses', 'goose': 'geese', 'library': 'libraries', 'factory': 'factories',
               'cherry': 'cherries', 'fairy': 'fairies', 'grandparent': 'grandparents', 'canoe': 'canoes',
               'taco': 'tacos', 'rhino': 'rhinos', 'hippo': 'hippos', 'flamingo': 'flamingos', 'mango': 'mangoes',
               'kangaroo': 'kangaroos', 'zoo': 'zoos', 'radio': 'radios'}
_EXTRA_MASS = """Germany Spain Greece Turkey Australia Argentina Peru Kenya Nigeria Norway Sweden Poland Ireland Scotland Korea
Vietnam Thailand Iran Israel Cuba Berlin Madrid Moscow Cairo Sydney Chicago Venice Hawaii Alaska Antarctica Mars Jupiter
lightning fog hail sand iron copper oil smoke ash dust mud grass moss autumn winter spring summer oxygen electricity gravity
happiness grief anxiety disgust gratitude nostalgia confusion panic relief trust courage kindness cruelty greed laziness
patience regret admiration embarrassment despair wonder sleepiness hunger thirst pain fatigue sarcasm humor honesty
deception ambition freedom peace justice chaos beauty economics physics chemistry biology astronomy geography architecture
literature theater cinema photography fashion cooking gardening fishing hunting camping travel education democracy
capitalism socialism taxes banking pollution recycling vaccines surgery nutrition fitness yoga meditation sleep memory
mythology magic Christmas Halloween slavery basketball tennis golf baseball swimming boxing skiing surfing poker opera jazz
ballet sculpture knitting origami garlic corn lettuce butter yogurt popcorn noodles curry steak bacon soda juice vinegar
salt sugar cinnamon vanilla candy soap""".split()
_EXTRA_PHRASED = [('Los Angeles', 'Los Angeles'), ('the Arctic', 'the Arctic'), ('the jungle', 'the jungle'),
                  ('the countryside', 'the countryside'), ('space', 'outer space'), ('the galaxy', 'the galaxy'),
                  ('the sky', 'the sky'), ('night', 'the night'), ('morning', 'the morning'), ('tides', 'the tides'),
                  ('the internet', 'the internet'), ('social media', 'social media'), ('video games', 'video games'),
                  ('climate change', 'climate change'), ('dreams', 'dreams'), ('the Bible', 'the Bible'),
                  ('the Roman Empire', 'the Roman Empire'), ('World War II', 'World War II'),
                  ('the Olympics', 'the Olympics'), ('rock music', 'rock music'), ('hip hop', 'hip hop'),
                  ('painting', 'painting'), ('ice cream', 'ice cream'), ('light bulbs', 'light bulbs'),
                  ('headphones', 'headphones'), ('scissors', 'scissors'), ('glasses', 'glasses'),
                  ('orange', 'the color orange'), ('brown', 'the color brown'), ('gray', 'the color gray'),
                  ('silver', 'the color silver'), ('turquoise', 'the color turquoise'), ('beige', 'the color beige')]
EXTRA = ([(w, _PLURAL_FIX.get(w, w + 's'), 'extra') for w in _EXTRA_PLURAL]
         + [(w, w, 'extra') for w in _EXTRA_MASS] + [(n, p, 'extra') for n, p in _EXTRA_PHRASED])

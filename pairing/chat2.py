"""The private items of the chat2 chats (prep_talk.py --chat2, game co_chat2): twelve kinds of item with hand-written
values and accepted words.

Each table line is value|alias,...: an item's accepted words are its value and 1 to 3 aliases. The first 20 values of
each training kind (TRAIN_KINDS) are training items and the rest test items; the held-out kinds (HELD_OUT_KINDS) have
test items only. FRAMES give each kind's sentence for the owner's prompt (second person) and for the partner's writer
prompt (third person, 'teacher').

Usage: import chat2; chat2.conversation('train', random.Random(0))
"""
import re

TRAIN_KINDS = ('city', 'job', 'pet', 'food', 'hobby', 'instrument', 'weather', 'event')
HELD_OUT_KINDS = ('mood', 'trip', 'book', 'sport')
TRAIN_VALUES = 20

CITY = """
Munich|Muenchen
Cologne|Koeln
Vienna|Wien
Rome|Roma
Florence|Firenze
Venice|Venezia
Milan|Milano
Naples|Napoli
Turin|Torino
Genoa|Genova
Lisbon|Lisboa
Athens|Athina
Prague|Praha
Warsaw|Warszawa
Krakow|Cracow
Beijing|Peking
Guangzhou|Canton
Mumbai|Bombay
Chennai|Madras
Kolkata|Calcutta
Bengaluru|Bangalore
Istanbul|Constantinople
Kyiv|Kiev
Busan|Pusan
Bangkok|Krungthep
Cairo|Qahira
"""
TABLES = {
    'city': CITY,
    'job': """
pilot|aviator
dentist|dentistry
plumber|plumbing
electrician|electrical
carpenter|carpentry
librarian|librarians
baker|bakers
chef|chefs
nurse|nursing
surgeon|surgery
teacher|teaching
firefighter|firefighting
paramedic|paramedics
mechanic|mechanics
farmer|farming
architect|architecture
journalist|journalism
photographer|photographers
accountant|accountancy
lawyer|attorney
translator|translating
veterinarian|vet
astronomer|astronomy
geologist|geology
florist|floristry
barber|barbers
""",
    'pet': """
dog|dogs
cat|cats
rabbit|bunny
hamster|hamsters
gerbil|gerbils
ferret|ferrets
chinchilla|chinchillas
hedgehog|hedgehogs
parrot|parrots
budgie|budgerigar
canary|canaries
cockatiel|cockatiels
finch|finches
dove|doves
turtle|turtles
tortoise|tortoises
gecko|geckos
iguana|iguanas
chameleon|chameleons
snake|snakes
goldfish|goldfishes
guppy|guppies
axolotl|axolotls
tarantula|tarantulas
frog|frogs
newt|newts
""",
    'food': """
pizza|pizzas
sushi|maki
lasagna|lasagne
tacos|taco
curry|curries
dumplings|dumpling
pancakes|pancake
waffles|waffle
omelette|omelet
risotto|risottos
falafel|falafels
hummus|houmous
paella|paellas
ramen|ramens
burrito|burritos
croissant|croissants
cheesecake|cheesecakes
brownie|brownies
bagel|bagels
pretzel|pretzels
pierogi|pierogies
samosa|samosas
gnocchi|gnoccho
ratatouille|ratatouilles
couscous|kuskus
schnitzel|schnitzels
""",
    'hobby': """
knitting|knit
crochet|crocheting
pottery|ceramics
origami|paperfolding
gardening|garden
birdwatching|birding
beekeeping|apiary
woodworking|woodwork
calligraphy|calligraphic
painting|paint
sketching|sketch
sculpture|sculpting
embroidery|embroidering
quilting|quilts
sewing|sew
weaving|weave
scrapbooking|scrapbook
journaling|journal
stargazing|stargaze
foraging|forage
geocaching|geocache
metalworking|metalwork
leatherworking|leatherwork
printmaking|printmaker
bookbinding|bookbinder
whittling|whittle
""",
    'instrument': """
piano|pianist
guitar|guitarist
violin|violinist
cello|cellist
viola|violist
flute|flautist
clarinet|clarinetist
oboe|oboist
bassoon|bassoonist
trumpet|trumpeter
trombone|trombonist
tuba|tubist
saxophone|sax
harp|harpist
banjo|banjoist
mandolin|mandolinist
ukulele|uke
accordion|accordionist
harmonica|harmonicas
xylophone|xylophonist
marimba|marimbas
sitar|sitarist
tabla|tablas
bagpipes|bagpipe
didgeridoo|didjeridu
balalaika|balalaikas
""",
    'weather': """
sunny|sunshine
rainy|rain
snowy|snow
foggy|fog
windy|wind
cloudy|clouds
stormy|storm
hailing|hail
sleeting|sleet
drizzly|drizzle
frosty|frost
icy|ice
humid|humidity
muggy|mugginess
sweltering|swelter
freezing|freeze
misty|mist
hazy|haze
smoggy|smog
blustery|bluster
balmy|balminess
arid|aridity
dusty|dust
thundery|thunder
squally|squalls
wintry|wintery
""",
    'mood': """
happy|happiness
sad|sadness
angry|anger
anxious|anxiety
excited|excitement
calm|calmness
lonely|loneliness
hopeful|hopefulness
nostalgic|nostalgia
grateful|gratitude
jealous|jealousy
bored|boredom
curious|curiosity
confused|confusion
proud|pride
ashamed|shame
relieved|relief
frustrated|frustration
inspired|inspiration
homesick|homesickness
optimistic|optimism
pessimistic|pessimism
restless|restlessness
embarrassed|embarrassment
contented|contentment
melancholy|melancholic
""",
    'trip': CITY,
    'book': """
dinosaurs|dinosaur
volcanoes|volcano
butterflies|butterfly
penguins|penguin
whales|whale
forests|forest
oceans|ocean
planets|planet
galaxies|galaxy
robots|robot
castles|castle
pirates|pirate
pharaohs|pharaoh
vikings|viking
samurai|samurais
dragons|dragon
wizards|wizard
witches|witch
fairies|fairy
ghosts|ghost
mushrooms|mushroom
fossils|fossil
minerals|mineral
insects|insect
bacteria|bacterium
viruses|virus
""",
    'sport': """
soccer|football
basketball|hoops
running|jogging
swimming|swam
volleyball|volleying
baseball|baseballs
cycling|bicycling
rugby|rugger
skiing|skied
lacrosse|lax
fencing|fenced
racquetball|racketball
golf|golfing
boxing|boxed
wrestling|wrestled
skating|skated
surfing|surfed
rowing|rowed
bocce|bocci
archery|bowmanship
curling|curlers
bowling|bowled
gymnastics|gymnast
judo|judoka
karate|karateka
sumo|sumotori
""",
}

# Events have explicit short owner/teacher sentences: no pronoun substitution.
EVENTS = """
wedding|married,marriage|Your brother just got married.|{P}'s brother just got married.
graduation|graduated|You just graduated from college.|{P} just graduated from college.
promotion|promoted|You just got promoted at work.|{P} just got promoted at work.
birthday|birthdays|You just celebrated your birthday.|{P} just celebrated a birthday.
engagement|engaged|You just got engaged.|{P} just got engaged.
retirement|retired|You just retired from work.|{P} just retired from work.
reunion|reunited|You just attended a family reunion.|{P} just attended a family reunion.
concert|concerts|You just attended a concert.|{P} just attended a concert.
festival|festivals|You just went to a festival.|{P} just went to a festival.
exhibition|exhibit|You just visited an art exhibition.|{P} just visited an art exhibition.
marathon|marathons|You just finished a marathon.|{P} just finished a marathon.
interview|interviewed|You just had a job interview.|{P} just had a job interview.
move|moved|You just moved into a new apartment.|{P} just moved into a new apartment.
adoption|adopted|You just adopted a child.|{P} just adopted a child.
breakup|split|You just went through a breakup.|{P} just went through a breakup.
funeral|funerals|You just attended a funeral.|{P} just attended a funeral.
recovery|recovered|You just recovered from an illness.|{P} just recovered from an illness.
award|awarded|You just received an award.|{P} just received an award.
rejection|rejected|Your application was just rejected.|{P}'s application was just rejected.
scholarship|scholarships|You just won a scholarship.|{P} just won a scholarship.
rehearsal|rehearsed|You just finished a theatre rehearsal.|{P} just finished a theatre rehearsal.
picnic|picnicked|You just had a picnic with friends.|{P} just had a picnic with friends.
renovation|renovated|You just renovated your kitchen.|{P} just renovated a kitchen.
resignation|resigned|You just resigned from your job.|{P} just resigned from a job.
audition|auditioned|You just auditioned for a play.|{P} just auditioned for a play.
anniversary|anniversaries|You just celebrated your anniversary.|{P} just celebrated an anniversary.
"""
FRAMES = {
    'city': ('Today you are in {v}.', '{P} is in {v} today.'),
    'job': ('You work as {a} {v}.', '{P} works as {a} {v}.'),
    'pet': ('You have {a} pet {v}.', '{P} has {a} pet {v}.'),
    'food': ('You just ate {v}, your favourite food.', '{P} just ate {v}, a favourite food.'),
    'hobby': ('Your hobby is {v}.', "{P}'s hobby is {v}."),
    'instrument': ('You are learning to play the {v}.', '{P} is learning to play the {v}.'),
    'weather': ('The weather where you are is {v}.', 'The weather where {P} is today is {v}.'),
    'mood': ('You feel {v} today.', '{P} feels {v} today.'),
    'trip': ('You are taking a trip to {v} next week.', '{P} is taking a trip to {v} next week.'),
    'book': ('You are reading a book about {v}.', '{P} is reading a book about {v}.'),
    'sport': ('You took part in {v} this morning.', '{P} took part in {v} this morning.'),
}


def _catalog():
    """{kind: [item]}, each item dict(kind, value, words, owner, teacher), from TABLES, FRAMES and EVENTS."""
    out = {}
    for kind, table in TABLES.items():
        out[kind] = []
        for line in table.strip().splitlines():
            value, aliases = line.split('|')
            # In the pet frame the article modifies 'pet', not the species.
            a = 'an' if kind == 'job' and value[0] in 'aeiou' else 'a'
            owner, teacher = [s.format(v=value, a=a, P='{P}') for s in FRAMES[kind]]
            out[kind].append(dict(kind=kind, value=value, words=[value] + aliases.split(','),
                                  owner=owner, teacher=teacher))
    out['event'] = [dict(kind='event', value=v, words=[v] + aliases.split(','), owner=own, teacher=teach)
                    for v, aliases, own, teach in (line.split('|') for line in EVENTS.strip().splitlines())]
    return out


CATALOG = _catalog()


def pool(kind, split):
    """The items of kind in split ('train' or 'test')."""
    if split not in ('train', 'test'):
        raise ValueError('chat2 split must be train or test')
    values = CATALOG[kind]
    if kind in HELD_OUT_KINDS:
        return [] if split == 'train' else values
    return values[:TRAIN_VALUES] if split == 'train' else values[TRAIN_VALUES:]


def scoring_words(item):
    """The accepted words of a private item: a chat2 item's words, or a co_chat city string as one word (none for
    None)."""
    return item['words'] if isinstance(item, dict) else [item] if item else []


def contains_item(text, item):
    """True when text contains one of the item's accepted words, as evalfree.contains matches them."""
    from evalfree import contains
    return any(contains(text, word) for word in scoring_words(item))


def kind_of(item):
    """The item's kind ('city' for a co_chat city string, None for no item)."""
    return item['kind'] if isinstance(item, dict) else 'city' if item else None


def kind_group(item):
    """'held_out_kind' or 'trained_kind' for the item's kind, None for no item."""
    k = kind_of(item)
    return ('held_out_kind' if k in HELD_OUT_KINDS else 'trained_kind') if k else None


def draw(split, rng):
    """A random item of split: a uniformly drawn kind (held-out kinds only in 'test'), then a uniformly drawn item."""
    kinds = TRAIN_KINDS if split == 'train' else TRAIN_KINDS + HELD_OUT_KINDS
    kind = rng.choice(kinds)
    row = rng.choice(pool(kind, split))
    return dict(row, words=list(row['words']))


def wording(split, rng):
    """A chat wording without its city sentence, since the private item gives the situation: CHAT_TEST for split
    'test', else one of the CHAT wordings."""
    from coread import CHAT, CHAT_TEST
    if split == 'test':
        return CHAT_TEST.replace('You are spending today in {city}. ', '')
    return rng.choice((CHAT[0].split('. ', 1)[1],
                       'A quiet moment with {P}. Talk about whatever comes up.',
                       CHAT[2].split('. ', 1)[1], *CHAT[3:]))


def conversation(split, rng):
    """A conversation for prep_talk.py --chat2: each model has a private item with probability 0.75; 'system' has the
    training prompts and 'teach' the writers' prompts, which add the partner's item."""
    from coread import chat_system
    from link import SIDES, partner
    opener = rng.choice(SIDES)
    private = {x: draw(split, rng) if rng.random() < .75 else None for x in SIDES}
    words = {x: wording(split, rng) for x in SIDES}
    return dict(game='chat2', opener=opener, private=private, asker=opener, holder=partner(opener), turns=[],
                system={x: chat_system(x, private[x], words[x]) for x in SIDES},
                teach={x: chat_system(x, private[x], words[x], private[partner(x)]) for x in SIDES})


def validate_catalog():
    """Asserts that every kind has at least 25 items, each with 2 to 4 single accepted words that no other item of the
    kind uses and that both of its sentences contain, and that the splits share no value; returns {kind: {split:
    number of items}}."""
    for kind, rows in CATALOG.items():
        assert len(rows) >= 25, kind
        seen = set()
        for row in rows:
            words = [w.casefold() for w in row['words']]
            assert 2 <= len(words) <= 4 and all(re.fullmatch(r'\w+', w) for w in words), row
            assert not seen.intersection(words), (kind, words)
            seen.update(words)
            assert contains_item(row['owner'], row) and contains_item(row['teacher'], row), row
        assert not {r['value'] for r in pool(kind, 'train')} & {r['value'] for r in pool(kind, 'test')}
    return {k: {s: len(pool(k, s)) for s in ('train', 'test')} for k in CATALOG}

import numpy as np
import random
from collections import Counter


male_nouns = ['n01045202', 'n02472293', 'n02472987', 'n02473720', 'n02473857', 'n02474431', 'n02475078', 'n02475478', 'n02475669', 'n02710201', 'n03716327', 'n04143712', 'n07182367', 'n07638574', 'n07989373', 'n09610405', 'n09637211', 'n09641002', 'n09643545', 'n09667205', 'n09761753', 'n09829923', 'n09851465', 'n09861395', 'n09871364', 'n09876308', 'n09882007', 'n09895480', 'n09902017', 'n09902731', 'n09924195', 'n09927451', 'n09937056', 'n09946437', 'n09954879', 'n10015792', 'n10050261', 'n10055181', 'n10056611', 'n10056719', 'n10058585', 'n10059323', 'n10077394', 'n10078333', 'n10090864', 'n10108089', 'n10113072', 'n10113583', 'n10120330', 'n10122645', 'n10133850', 'n10144338', 'n10152083', 'n10159289', 'n10161867', 'n10162016', 'n10168457', 'n10176679', 'n10180923', 'n10193967', 'n10216560', 'n10223744', 'n10240514', 'n10251517', 'n10271216', 'n10287213', 'n10288516', 'n10288763', 'n10289039', 'n10289176', 'n10291730', 'n10291822', 'n10291942', 'n10292052', 'n10305802', 'n10309496', 'n10314054', 'n10321754', 'n10336904', 'n10375214', 'n10375402', 'n10375506', 'n10375690', 'n10383094', 'n10403633', 'n10433610', 'n10433737', 'n10436707', 'n10448065', 'n10448157', 'n10464178', 'n10469611', 'n10483395', 'n10513120', 'n10519884', 'n10520109', 'n10520286', 'n10521100', 'n10528148', 'n10531109', 'n10546850', 'n10574425', 'n10575089', 'n10582746', 'n10625285', 'n10626439', 'n10639359', 'n10639469', 'n10643476', 'n10648237', 'n10660883', 'n10722758', 'n10742881', 'n10742997', 'n10745332', 'n10753339', 'n10781684', 'n10791221', 'n10804287', 'n11929477', 'n12025507', 'n15230482']
female_nouns = ['n08477634', 'n09637339', 'n09641130', 'n09643670', 'n09772930', 'n09911226', 'n09941172', 'n10058962', 'n10096126', 'n10129825', 'n10183347', 'n10189776', 'n10254965', 'n10270468', 'n10322648', 'n10323752', 'n10377021', 'n10448455', 'n10485440', 'n10609198', 'n10739512', 'n10739636', 'n10780284', 'n10780632', 'n10787197', 'n10787470', 'n10788852', 'n10789820', 'n11931312']
interval_words = [
    ['guarding', 'plowing', 'exterminating', 'ejecting', 'welding', 'prying', 'repairing', 'apprehending', 'installing', 'constructing', 'detaining', 'docking', 'hunting', 'baptizing', 'building', 'destroying', 'preaching', 'clearing', 'spearing', 'sharpening', 'barbecuing', 'fixing', 'plummeting', 'subduing', 'mowing', 'parachuting', 'camouflaging', 'chiseling', 'urinating', 'pitching', 'tuning', 'colliding', 'unloading', 'deflecting', 'handcuffing', 'fueling', 'grilling', 'coaching', 'shearing', 'carving', 'burning', 'flinging', 'nailing', 'attacking', 'interrogating', 'shaking', 'loading', 'brewing', 'tipping', 'panhandling', 'tilling', 'piloting', 'hoisting', 'hurling', 'filming', 'flexing', 'scoring', 'fishing', 'carting', 'breaking', 'tattooing', 'fording', 'drumming', 'dousing', 'sealing', 'officiating', 'bowing', 'unveiling', 'juggling', 'whipping', 'extinguishing', 'assembling', 'shoveling', 'catching', 'trimming', 'betting', 'punching', 'spraying'], 
    ['unplugging', 'harvesting', 'operating', 'striking', 'knocking', 'spitting', 'surfing', 'grinding', 'frisking', 'burying', 'clapping', 'plunging', 'offering', 'ducking', 'throwing', 'peeing', 'skiing', 'slicing', 'aiming', 'dragging', 'disembarking', 'shooting', 'shaving', 'crawling', 'videotaping', 'hitting', 'camping', 'saying', 'unlocking', 'igniting', 'chasing', 'ascending', 'signing', 'scraping', 'lecturing', 'dropping', 'biking', 'mending', 'climbing', 'clenching', 'fastening', 'flipping', 'descending', 'slipping', 'lifting', 'painting', 'recording', 'brawling', 'uncorking', 'grimacing', 'pedaling', 'falling', 'tearing', 'spreading', 'pruning', 'steering', 'restraining', 'molding', 'applauding', 'pulling', 'stripping', 'adjusting', 'speaking', 'glaring', 'smashing', 'saluting', 'training', 'confronting', 'shredding', 'shouting', 'kissing', 'chopping', 'autographing', 'crushing', 'begging', 'rowing', 'standing', 'inflating', 'sketching', 'staring', 'waving', 'jumping', 'sprinting', 'filling', 'squinting', 'photographing', 'distributing', 'pumping', 'attaching', 'heaving', 'kicking', 'signaling', 'lighting', 'squeezing', 'sowing', 'leading', 'stumbling', 'frowning', 'wading', 'raking', 'miming', 'competing', 'placing', 'spilling', 'hoeing', 'voting', 'emptying', 'tying', 'leaping', 'wheeling', 'recuperating', 'taping', 'pouring', 'performing', 'hiking', 'stapling', 'recovering', 'buttoning', 'planting', 'stacking', 'diving', 'dripping', 'clipping', 'kneeling', 'tugging', 'running', 'crouching', 'inserting', 'driving', 'whistling', 'gambling', 'displaying', 'poking', 'drenching', 'tripping', 'working', 'pressing', 'nuzzling', 'spying', 'hunching', 'sliding', 'laughing'],
    ['paying', 'immersing', 'stooping', 'skating', 'rehabilitating', 'grinning', 'sweeping', 'praying', 'measuring', 'scratching', 'submerging', 'dialing', 'shrugging', 'frying', 'pushing', 'packaging', 'cramming', 'checking', 'sniffing', 'pooing', 'bothering', 'injecting', 'slouching', 'turning', 'opening', 'riding', 'sitting', 'dining', 'locking', 'giving', 'instructing', 'yanking', 'embracing', 'snuggling', 'coughing', 'leaning', 'caressing', 'socializing', 'gluing', 'yawning', 'lathering', 'carrying', 'walking', 'spanking', 'communicating', 'wagging', 'rubbing', 'interviewing', 'boarding', 'cleaning', 'drawing', 'perspiring', 'sleeping', 'tickling', 'vaulting', 'celebrating', 'stirring', 'slapping', 'hitchhiking', 'putting', 'resting', 'wetting', 'licking', 'asking', 'wiping', 'examining', 'swimming', 'winking', 'weeding', 'bandaging', 'pinning', 'gnawing', 'vacuuming', 'phoning', 'complaining', 'tilting', 'foraging', 'pricking', 'peeling', 'strapping', 'stroking', 'writing', 'stuffing', 'encouraging', 'sprinkling', 'crafting', 'talking', 'helping', 'drinking', 'shelving', 'wringing', 'dissecting', 'splashing', 'pouting', 'patting', 'mopping', 'releasing', 'buckling', 'milking', 'shushing', 'floating', 'bouncing', 'counting', 'clinging', 'unpacking', 'gasping', 'serving', 'crying', 'ignoring', 'watering', 'making', 'waiting', 'covering', 'lacing', 'eating', 'feeding', 'practicing', 'admiring', 'gardening', 'studying', 'reading', 'shivering', 'weighing', 'pasting', 'distracting', 'hugging', 'scooping', 'singing', 'reassuring'],
    ['typing', 'tasting', 'wrapping', 'emerging', 'jogging', 'whisking', 'calling', 'providing', 'packing', 'smiling', 'imitating', 'swinging', 'biting', 'buying', 'baking', 'scrubbing', 'massaging', 'telephoning', 'disciplining', 'browsing', 'dipping', 'potting', 'cooking', 'misbehaving', 'moistening', 'scolding', 'erasing', 'sucking', 'washing', 'weeping', 'pinching', 'kneading', 'mashing', 'chewing', 'decorating', 'rocking', 'picking', 'smearing', 'twirling', 'smelling', 'dyeing', 'rinsing', 'sneezing', 'hanging', 'folding', 'crowning', 'twisting', 'stretching', 'exercising', 'wrinkling', 'soaking', 'flossing', 'skipping', 'dancing', 'stitching', 'puckering', 'drying', 'microwaving', 'shelling', 'shopping', 'grieving', 'applying', 'brushing', 'combing', 'buttering'],
    ['giggling', 'dusting', 'arching', 'arranging', 'bathing', 'fetching', 'sewing', 'spinning', 'coloring', 'calming', 'nagging', 'waxing', 'moisturizing', 'braiding', 'curling', 'manicuring', 'curtsying', 'cheerleading']
]


def rand_split_test_set(dataset, test_set_imgs_per_class):
    verbs = np.unique([x.split('_')[0] for x in dataset.keys()])
    images = list(dataset.keys())
    random.shuffle(images)
    train_images, test_images = [], []
    for verb in verbs:
        matching_images = [name for name in images if verb == name.split('_')[0]]
        train_images += matching_images[:-test_set_imgs_per_class]
        test_images += matching_images[-test_set_imgs_per_class:]
    train_set = {image: dataset[image] for image in train_images}
    test_set = {image: dataset[image] for image in test_images}
    return test_set, train_set

def rand_split_dataset(dataset, partition_size):
    images = list(dataset.keys())
    random.shuffle(images)
    first_set = {image: dataset[image] for image in images[:partition_size]}
    second_set = {image: dataset[image] for image in images[partition_size:]}
    return first_set, second_set


def collapse_annotations(dataset, use_majority=True):
    dataset_collapsed = {}
    for image, annotations in dataset.items():
        frame_collapsed = {k: [] for k in annotations['frames'][0].keys()}
        for frame in annotations['frames']:
            for k, v in frame.items():
                if v.strip():
                    frame_collapsed[k].append(v)
        for k in frame_collapsed.keys():
            if len(frame_collapsed[k]) == 0:
                frame_collapsed[k] = ['']
            maj_v, count = Counter(frame_collapsed[k]).most_common(1)[0]
            if use_majority and count > 1:
                frame_collapsed[k] = maj_v
            else:
                frame_collapsed[k] = random.choice(frame_collapsed[k])
        dataset_collapsed[image] = {'verb': annotations['verb'], 'frames': [frame_collapsed]}
    return dataset_collapsed


def transform_preds_to_dataset(preds):
    dataset = {}
    for img_name, info in preds.items():
        info2 = {'verb': info[0]['verb'], 'frames': info[0]['frames']}
        dataset[img_name + '.jpg'] = info2
    return dataset


def get_dataset_gender_stats(dataset):
    data_interval_ratios = [{'man': 0, 'woman': 0} for _ in range(len(interval_words))]
    cooking_man, cooking_woman = 0, 0

    for _, info in dataset.items():
        agents = [frame['agent'] for frame in info['frames'] if 'agent' in frame]
        if len(agents) == 0: continue
        is_man = any(noun in agents for noun in male_nouns)
        is_woman = any(noun in agents for noun in female_nouns)
        for j, words in enumerate(interval_words):
            if info['verb'] in words:
                if is_man and not is_woman: data_interval_ratios[j]['man'] += 1
                if is_woman and not is_man: data_interval_ratios[j]['woman'] += 1
        if info['verb'] == 'cooking':
            if is_man and not is_woman: cooking_man += 1
            if is_woman and not is_man: cooking_woman += 1

    total_man, total_woman = sum([x['man'] for x in data_interval_ratios]), sum([x['woman'] for x in data_interval_ratios])
    dataset_ratio = total_woman / (total_woman + total_man + np.finfo(float).eps)
    dataset_interval_ratios = [ratio['woman']/(ratio['man']+ratio['woman']+np.finfo(float).eps) for ratio in data_interval_ratios]
    cooking_ratio = cooking_woman / (cooking_man + cooking_woman + np.finfo(float).eps)
    
    return {'overall_ratio': dataset_ratio, 'cooking_ratio': cooking_ratio} | \
           {f'interval_{i}_ratio': ratio for i, ratio in enumerate(dataset_interval_ratios)}

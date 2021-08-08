PAGE_TITLE = 'Car part Classifier'
PAGE_ICON = "🤖"
LAYOUT = "wide"

LANDINGPAGE_TITLE = 'Car part Classification'
SIDEBAR_TITLE = '✨ Navigation ✨'


CONTENT_IMAGES_FILE = [
    'tail_light.jpg', 'wheel.jpg', 'steering_wheel.jpg',
    'head_light.jpg', 'rear_view_mirror.jpg', 'side_mirror.jpg',
]
CONTENT_IMAGES_NAME = [
    'Tail light', 'Wheel', 'Steering wheel',
    'Head light', 'Rear view mirror', 'Side mirror',
]

IMAGES_PATH = 'images'

DEFAULT_CONFIDENCE_THRESHOLD = 0.8

VIDEO_CONFIDENCE = 0.8

MEAN = [[[0.485 ]], [[0.456 ]], [[0.4065]]]
STD = [[[0.229]], [[0.224]], [[0.225]]]

import random
from collections import Counter, defaultdict
from enum import Enum, unique

import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder


@unique
class Normalisation(str, Enum):
    NONE = 'no_norm'
    ROI = 'roi_norm'
    SUBJECT = 'subject_norm'


@unique
class ConnType(str, Enum):
    FMRI = 'fmri'
    STRUCT = 'struct'


@unique
class ConvStrategy(str, Enum):
    CNN_ENTIRE = 'entire'
    TCN_ENTIRE = 'tcn_entire'


@unique
class PoolingStrategy(str, Enum):
    MEAN = 'mean'
    DIFFPOOL = 'diff_pool'
    CONCAT = 'concat'

@unique
class AnalysisType(str, Enum):
    SPATIOTEMOPRAL = 'spatiotemporal'
    FLATTEN_CORRS = 'flatten_corrs'
    FLATTEN_CORRS_THRESHOLD = 'flatten_corrs_threshold'

@unique
class EncodingStrategy(str, Enum):
    NONE = 'none'
    AE3layers = '3layerAE'


NEW_STRUCT_PEOPLE = [100206, 100307, 100408, 100610, 101107, 101309, 101410, 101915, 102008, 102311, 102513, 102614,
                     102715, 102816, 103010, 103111, 103212, 103414, 103515, 103818, 104012, 104416, 104820, 105014,
                     105115, 105216, 105620, 105923, 106016, 106319, 106521, 106824, 107018, 107321, 107422, 107725,
                     108020, 108121, 108222, 108323, 108525, 108828, 109123, 109830, 110007, 110411, 110613, 111009,
                     111211, 111312, 111413, 111514, 111716, 112112, 112314, 112516, 112819, 112920, 113215, 113316,
                     113619, 113821, 113922, 114116, 114217, 114318, 114419, 114621, 114823, 115017, 115219, 115320,
                     115724, 115825, 116221, 116423, 116524, 116726, 117021, 117122, 117324, 117930, 118023, 118124,
                     118225, 118528, 118730, 118831, 118932, 119025, 119126, 119732, 119833, 120010, 120111, 120212,
                     120414, 120515, 120717, 121416, 121618, 121719, 121921, 122317, 122418, 122620, 122822, 123117,
                     123420, 123521, 123723, 123824, 123925, 124220, 124422, 124624, 124826, 125222, 125424, 125525,
                     126325, 126426, 126628, 127226, 127327, 127630, 127731, 127832, 127933, 128026, 128127, 128632,
                     128935, 129028, 129129, 129331, 129634, 129937, 130013, 130114, 130316, 130417, 130518, 130619,
                     130720, 130821, 130922, 131217, 131419, 131722, 131823, 131924, 132017, 132118, 133019, 133625,
                     133827, 133928, 134021, 134223, 134324, 134425, 134627, 134728, 134829, 135124, 135225, 135528,
                     135629, 135730, 135932, 136126, 136227, 136631, 136732, 136833, 137027, 137128, 137229, 137431,
                     137532, 137633, 137936, 138130, 138231, 138332, 138534, 138837, 139233, 139435, 139637, 139839,
                     140117, 140319, 140420, 140824, 140925, 141119, 141422, 141826, 142828, 143224, 143325, 143426,
                     143830, 144125, 144226, 144428, 144731, 144832, 144933, 145127, 145632, 145834, 146129, 146331,
                     146432, 146533, 146634, 146735, 146836, 146937, 147030, 147636, 147737, 148032, 148133, 148335,
                     148436, 148840, 148941, 149236, 149337, 149539, 149741, 149842, 150019, 150524, 150625, 150726,
                     150928, 151021, 151223, 151324, 151425, 151526, 151627, 151728, 151829, 151930, 152225, 152427,
                     152831, 153025, 153126, 153227, 153429, 153631, 153732, 153833, 153934, 154229, 154330, 154431,
                     154532, 154734, 154835, 154936, 155231, 155635, 155938, 156031, 156233, 156334, 156435, 156536,
                     156637, 157336, 157437, 157942, 158035, 158136, 158338, 158540, 158843, 159138, 159239, 159340,
                     159744, 159946, 160123, 160729, 160830, 160931, 161327, 161630, 161731, 161832, 162026, 162228,
                     162329, 162733, 162935, 163129, 163331, 163432, 163836, 164030, 164131, 164636, 164939, 165032,
                     165436, 165638, 165840, 165941, 166438, 166640, 167036, 167238, 167440, 167743, 168139, 168240,
                     168341, 168745, 168947, 169040, 169343, 169444, 169545, 169747, 169949, 170631, 170934, 171330,
                     171431, 171532, 171633, 172029, 172130, 172332, 172433, 172534, 172635, 172938, 173132, 173233,
                     173334, 173435, 173536, 173637, 173738, 173839, 173940, 174437, 174841, 175035, 175136, 175237,
                     175338, 175439, 175540, 175742, 176037, 176239, 176441, 176542, 176744, 176845, 177140, 177241,
                     177342, 177645, 177746, 178142, 178243, 178647, 178748, 178849, 178950, 179245, 179346, 179952,
                     180129, 180230, 180432, 180533, 180735, 180836, 180937, 181131, 181232, 181636, 182032, 182436,
                     182739, 182840, 183034, 183337, 183741, 185038, 185139, 185341, 185442, 185846, 185947, 186040,
                     186141, 186444, 186545, 186848, 187143, 187345, 187547, 187850, 188145, 188347, 188448, 188549,
                     188751, 189349, 189450, 189652, 190031, 191033, 191235, 191336, 191437, 191841, 191942, 192035,
                     192136, 192237, 192439, 192540, 192641, 192843, 193239, 193441, 193845, 194140, 194443, 194645,
                     194746, 194847, 195041, 195445, 195647, 195849, 195950, 196144, 196346, 196750, 196851, 196952,
                     197348, 197550, 197651, 198047, 198249, 198350, 198451, 198653, 198855, 199150, 199251, 199352,
                     199453, 199655, 199958, 200008, 200109, 200210, 200311, 200513, 200614, 200917, 201111, 201414,
                     201515, 201818, 202113, 202719, 202820, 203418, 203923, 204016, 204218, 204319, 204420, 204521,
                     204622, 205119, 205220, 205725, 205826, 206222, 206323, 206525, 206727, 206828, 206929, 207123,
                     207426, 208024, 208125, 208226, 208327, 208428, 208630, 209127, 209228, 209329, 209834, 209935,
                     210011, 210112, 210415, 210617, 211114, 211215, 211316, 211417, 211619, 211720, 211821, 211922,
                     212015, 212116, 212217, 212318, 212419, 212823, 213017, 213421, 213522, 214019, 214221, 214423,
                     214524, 214625, 214726, 217126, 217429, 219231, 220721, 221319, 223929, 224022, 227432, 227533,
                     228434, 231928, 233326, 236130, 237334, 238033, 239944, 245333, 246133, 248238, 248339, 249947,
                     250427, 250932, 251833, 255639, 255740, 256540, 257542, 257845, 257946, 263436, 268749, 268850,
                     270332, 274542, 275645, 280739, 280941, 281135, 283543, 284646, 285345, 285446, 286347, 286650,
                     287248, 289555, 290136, 293748, 295146, 297655, 298051, 298455, 299154, 299760, 300618, 300719,
                     303119, 303624, 304020, 304727, 305830, 307127, 308129, 308331, 309636, 310621, 311320, 314225,
                     316633, 316835, 317332, 318637, 320826, 321323, 322224, 325129, 329440, 329844, 330324, 333330,
                     334635, 336841, 339847, 341834, 342129, 346137, 346945, 348545, 349244, 350330, 351938, 352132,
                     352738, 353740, 355239, 356948, 358144, 360030, 361234, 361941, 362034, 365343, 366042, 366446,
                     368551, 368753, 371843, 376247, 377451, 378756, 378857, 379657, 380036, 381038, 381543, 382242,
                     385046, 385450, 386250, 387959, 389357, 390645, 391748, 392447, 392750, 393247, 393550, 394956,
                     395251, 395756, 395958, 397154, 397760, 397861, 406432, 406836, 412528, 414229, 415837, 422632,
                     424939, 429040, 432332, 433839, 436239, 436845, 441939, 445543, 448347, 449753, 453441, 456346,
                     459453, 465852, 467351, 473952, 475855, 479762, 480141, 481951, 485757, 486759, 492754, 495255,
                     497865, 499566, 500222, 506234, 510326, 512835, 513736, 517239, 519950, 520228, 521331, 522434,
                     523032, 524135, 525541, 529549, 529953, 530635, 531536, 536647, 540436, 541943, 545345, 547046,
                     548250, 549757, 553344, 555348, 555651, 557857, 559053, 561242, 561444, 562345, 562446, 565452,
                     566454, 567052, 567961, 568963, 570243, 571144, 571548, 572045, 573249, 573451, 576255, 579665,
                     579867, 580044, 580347, 580650, 580751, 581349, 581450, 583858, 585256, 585862, 586460, 587664,
                     588565, 592455, 594156, 597869, 598568, 599065, 599469, 599671, 601127, 604537, 609143, 611938,
                     613538, 614439, 615744, 616645, 617748, 618952, 620434, 622236, 623844, 626648, 627549, 627852,
                     628248, 633847, 638049, 644044, 645450, 645551, 647858, 654350, 654754, 656253, 656657, 657659,
                     660951, 663755, 664757, 665254, 667056, 668361, 671855, 672756, 673455, 677766, 677968, 679568,
                     679770, 680250, 680957, 683256, 685058, 686969, 687163, 690152, 693461, 693764, 695768, 700634,
                     702133, 704238, 705341, 706040, 707749, 709551, 713239, 715041, 715647, 715950, 720337, 724446,
                     725751, 727553, 727654, 729254, 729557, 731140, 732243, 734045, 735148, 737960, 742549, 744553,
                     748258, 749058, 749361, 751348, 751550, 753150, 753251, 756055, 759869, 761957, 765056, 766563,
                     767464, 769064, 770352, 771354, 773257, 779370, 782561, 783462, 784565, 786569, 788876, 789373,
                     792564, 792766, 792867, 793465, 800941, 802844, 803240, 810439, 810843, 812746, 814649, 816653,
                     818859, 820745, 825048, 826353, 826454, 833148, 833249, 835657, 837560, 837964, 841349, 843151,
                     844961, 845458, 849264, 849971, 852455, 856463, 856766, 856968, 857263, 859671, 861456, 865363,
                     867468, 870861, 871762, 871964, 872158, 872562, 872764, 873968, 877168, 877269, 880157, 882161,
                     885975, 887373, 889579, 891667, 894067, 894673, 894774, 896778, 896879, 898176, 899885, 901038,
                     901139, 901442, 904044, 907656, 910241, 910443, 912447, 917255, 917558, 919966, 922854, 923755,
                     930449, 932554, 937160, 947668, 951457, 952863, 955465, 957974, 958976, 959574, 965367, 965771,
                     966975, 972566, 978578, 979984, 983773, 984472, 987983, 990366, 991267, 992673, 992774, 993675,
                     994273, 996782]

NEW_MULTIMODAL_TIMESERIES = [100206, 100307, 100408, 100610, 101006, 101107, 101309, 101410, 101915, 102008, 102109,
                             102311, 102513, 102614, 102715, 102816, 103010, 103111, 103212, 103414, 103515, 103818,
                             104012, 104416, 104820, 105014, 105115, 105216, 105620, 105923, 106016, 106319, 106521,
                             106824, 107018, 107220, 107321, 107422, 107725, 108020, 108121, 108222, 108323, 108525,
                             108828, 109123, 109325, 109830, 110007, 110411, 110613, 111009, 111211, 111312, 111413,
                             111514, 111716, 112112, 112314, 112516, 112819, 112920, 113215, 113316, 113417, 113619,
                             113821, 113922, 114116, 114217, 114318, 114419, 114621, 114823, 114924, 115017, 115219,
                             115320, 115724, 115825, 116120, 116221, 116423, 116524, 116726, 117021, 117122, 117324,
                             117728, 117930, 118023, 118124, 118225, 118528, 118730, 118831, 118932, 119025, 119126,
                             119732, 119833, 120010, 120111, 120212, 120414, 120515, 120717, 121315, 121416, 121618,
                             121719, 121820, 121921, 122317, 122418, 122620, 122822, 123117, 123420, 123521, 123723,
                             123824, 123925, 124220, 124422, 124624, 124826, 125222, 125424, 125525, 126325, 126426,
                             126628, 127226, 127327, 127630, 127731, 127832, 127933, 128026, 128127, 128329, 128632,
                             128935, 129028, 129129, 129331, 129533, 129634, 129937, 130013, 130114, 130316, 130417,
                             130518, 130619, 130720, 130821, 130922, 131217, 131419, 131722, 131823, 131924, 132017,
                             132118, 133019, 133625, 133827, 133928, 134021, 134223, 134324, 134425, 134627, 134728,
                             134829, 135124, 135225, 135528, 135629, 135730, 135932, 136126, 136227, 136631, 136732,
                             136833, 137027, 137128, 137229, 137431, 137532, 137633, 137936, 138130, 138231, 138332,
                             138534, 138837, 139233, 139435, 139637, 139839, 140117, 140319, 140420, 140824, 140925,
                             141119, 141422, 141826, 142424, 142828, 143224, 143325, 143426, 143830, 144125, 144226,
                             144428, 144731, 144832, 144933, 145127, 145531, 145632, 145834, 146129, 146331, 146432,
                             146533, 146634, 146735, 146836, 146937, 147030, 147636, 147737, 148032, 148133, 148335,
                             148436, 148840, 148941, 149236, 149337, 149539, 149741, 149842, 150019, 150423, 150524,
                             150625, 150726, 150928, 151021, 151223, 151324, 151425, 151526, 151627, 151728, 151829,
                             151930, 152225, 152427, 152831, 153025, 153126, 153227, 153429, 153631, 153732, 153833,
                             153934, 154229, 154330, 154431, 154532, 154734, 154835, 154936, 155231, 155635, 155938,
                             156031, 156233, 156334, 156435, 156536, 156637, 157336, 157437, 157942, 158035, 158136,
                             158338, 158540, 158843, 159138, 159239, 159340, 159441, 159744, 159845, 159946, 160123,
                             160729, 160830, 160931, 161327, 161630, 161731, 161832, 162026, 162228, 162329, 162733,
                             162935, 163129, 163331, 163432, 163836, 164030, 164131, 164636, 164939, 165032, 165234,
                             165436, 165638, 165840, 165941, 166438, 166640, 167036, 167238, 167440, 167743, 168139,
                             168240, 168341, 168745, 168947, 169040, 169141, 169343, 169444, 169545, 169747, 169949,
                             170631, 170934, 171128, 171330, 171431, 171532, 171633, 171734, 172029, 172130, 172332,
                             172433, 172534, 172635, 172938, 173132, 173233, 173334, 173435, 173536, 173637, 173738,
                             173839, 173940, 174437, 174841, 175035, 175136, 175237, 175338, 175439, 175540, 175742,
                             176037, 176239, 176441, 176542, 176744, 176845, 177140, 177241, 177342, 177645, 177746,
                             178142, 178243, 178647, 178748, 178849, 178950, 179245, 179346, 179952, 180129, 180230,
                             180432, 180533, 180735, 180836, 180937, 181131, 181232, 181636, 182032, 182436, 182739,
                             182840, 183034, 183337, 183741, 185038, 185139, 185341, 185442, 185846, 185947, 186040,
                             186141, 186444, 186545, 186848, 186949, 187143, 187345, 187547, 187850, 188145, 188347,
                             188448, 188549, 188751, 189349, 189450, 189652, 190031, 190132, 191033, 191235, 191336,
                             191437, 191841, 191942, 192035, 192136, 192237, 192439, 192540, 192641, 192843, 193239,
                             193441, 193845, 194140, 194443, 194645, 194746, 194847, 195041, 195445, 195647, 195849,
                             195950, 196144, 196346, 196750, 196851, 196952, 197348, 197550, 197651, 198047, 198249,
                             198350, 198451, 198653, 198855, 199150, 199251, 199352, 199453, 199655, 199958, 200008,
                             200109, 200210, 200311, 200513, 200614, 200917, 201111, 201414, 201515, 201717, 201818,
                             202113, 202719, 202820, 203418, 203923, 204016, 204218, 204319, 204420, 204521, 204622,
                             205119, 205220, 205725, 205826, 206222, 206323, 206525, 206727, 206828, 206929, 207123,
                             207426, 208024, 208125, 208226, 208327, 208428, 208630, 209127, 209228, 209329, 209531,
                             209834, 209935, 210011, 210112, 210415, 210617, 211114, 211215, 211316, 211417, 211619,
                             211720, 211821, 211922, 212015, 212116, 212217, 212318, 212419, 212823, 213017, 213421,
                             213522, 214019, 214221, 214423, 214524, 214625, 214726, 217126, 217429, 219231, 220721,
                             221218, 221319, 223929, 224022, 227432, 227533, 228434, 231928, 233326, 236130, 237334,
                             238033, 239136, 239944, 245333, 246133, 248238, 248339, 249947, 250427, 250932, 251833,
                             255639, 255740, 256540, 257542, 257845, 257946, 263436, 268749, 268850, 270332, 274542,
                             275645, 280739, 280941, 281135, 283543, 284646, 285345, 285446, 286347, 286650, 287248,
                             289555, 290136, 293748, 295146, 297655, 298051, 298455, 299154, 299760, 300618, 300719,
                             303119, 303624, 304020, 304727, 305830, 307127, 308129, 308331, 309636, 310621, 311320,
                             314225, 316633, 316835, 317332, 318637, 320826, 321323, 322224, 325129, 329440, 329844,
                             330324, 333330, 334635, 336841, 339847, 341834, 342129, 346137, 346945, 348545, 349244,
                             350330, 351938, 352132, 352738, 353740, 355239, 356948, 358144, 360030, 361234, 361941,
                             362034, 365343, 366042, 366446, 368551, 368753, 371843, 376247, 377451, 378756, 378857,
                             379657, 380036, 381038, 381543, 382242, 385046, 385450, 386250, 387959, 389357, 390645,
                             391748, 392447, 392750, 393247, 393550, 394956, 395251, 395756, 395958, 397154, 397760,
                             397861, 401422, 406432, 406836, 412528, 413934, 414229, 415837, 419239, 421226, 422632,
                             424939, 429040, 432332, 433839, 436239, 436845, 441939, 445543, 448347, 449753, 453441,
                             453542, 454140, 456346, 459453, 461743, 462139, 463040, 465852, 467351, 468050, 469961,
                             473952, 475855, 479762, 480141, 481042, 481951, 485757, 486759, 492754, 495255, 497865,
                             499566, 500222, 506234, 510225, 510326, 512835, 513130, 513736, 516742, 517239, 518746,
                             519647, 519950, 520228, 521331, 522434, 523032, 524135, 525541, 529549, 529953, 530635,
                             531536, 531940, 536647, 540436, 541640, 541943, 545345, 547046, 548250, 549757, 550439,
                             552241, 552544, 553344, 555348, 555651, 555954, 557857, 558657, 558960, 559053, 559457,
                             561242, 561444, 561949, 562345, 562446, 565452, 566454, 567052, 567759, 567961, 568963,
                             569965, 570243, 571144, 571548, 572045, 573249, 573451, 576255, 578057, 578158, 579665,
                             579867, 580044, 580347, 580650, 580751, 581349, 581450, 583858, 585256, 585862, 586460,
                             587664, 588565, 589567, 590047, 592455, 594156, 597869, 598568, 599065, 599469, 599671,
                             601127, 604537, 609143, 611938, 613235, 613538, 614439, 615441, 615744, 616645, 617748,
                             618952, 620434, 622236, 623137, 623844, 626648, 627549, 627852, 628248, 633847, 634748,
                             635245, 638049, 644044, 644246, 645450, 645551, 647858, 654350, 654552, 654754, 656253,
                             656657, 657659, 660951, 662551, 663755, 664757, 665254, 667056, 668361, 671855, 672756,
                             673455, 675661, 677766, 677968, 679568, 679770, 680250, 680452, 680957, 683256, 685058,
                             686969, 687163, 688569, 689470, 690152, 692964, 693461, 693764, 694362, 695768, 698168,
                             700634, 701535, 702133, 704238, 705341, 706040, 707749, 709551, 713239, 715041, 715647,
                             715950, 720337, 723141, 724446, 725751, 727553, 727654, 728454, 729254, 729557, 731140,
                             732243, 734045, 734247, 735148, 737960, 742549, 744553, 748258, 748662, 749058, 749361,
                             751348, 751550, 753150, 753251, 756055, 757764, 759869, 760551, 761957, 763557, 765056,
                             765864, 766563, 767464, 769064, 770352, 771354, 773257, 774663, 779370, 782561, 783462,
                             784565, 786569, 788674, 788876, 789373, 792564, 792766, 792867, 793465, 800941, 802844,
                             803240, 804646, 809252, 810439, 810843, 812746, 814548, 814649, 815247, 816653, 818455,
                             818859, 820745, 822244, 825048, 825553, 825654, 826353, 826454, 827052, 828862, 832651,
                             833148, 833249, 835657, 837560, 837964, 841349, 843151, 844961, 845458, 849264, 849971,
                             852455, 856463, 856766, 856968, 857263, 859671, 861456, 865363, 867468, 869472, 870861,
                             871762, 871964, 872158, 872562, 872764, 873968, 877168, 877269, 878776, 878877, 880157,
                             882161, 884064, 885975, 886674, 887373, 888678, 889579, 891667, 894067, 894673, 894774,
                             896778, 896879, 898176, 899885, 901038, 901139, 901442, 902242, 904044, 905147, 907656,
                             908860, 910241, 910443, 911849, 912447, 917255, 917558, 919966, 922854, 923755, 926862,
                             927359, 929464, 930449, 932554, 933253, 937160, 942658, 943862, 947668, 951457, 952863,
                             953764, 955465, 957974, 958976, 959574, 962058, 965367, 965771, 966975, 969476, 970764,
                             971160, 972566, 973770, 978578, 979984, 983773, 984472, 987074, 987983, 989987, 990366,
                             991267, 992673, 992774, 993675, 994273, 995174, 996782]

OLD_NETMATS_PEOPLE = [100206, 100307, 100408, 100610, 101006, 101107, 101309, 101915, 102008, 102109, 102311, 102513,
                      102614, 102715, 102816, 103010, 103111, 103212, 103414, 103515, 103818, 104012, 104416, 104820,
                      105014, 105115, 105216, 105620, 105923, 106016, 106319, 106521, 106824, 107018, 107321, 107422,
                      107725, 108020, 108121, 108222, 108323, 108525, 108828, 109123, 109325, 109830, 110007, 110411,
                      110613, 111009, 111211, 111312, 111413, 111514, 111716, 112112, 112314, 112516, 112920, 113215,
                      113316, 113619, 113922, 114217, 114318, 114419, 114621, 114823, 114924, 115017, 115219, 115320,
                      115724, 115825, 116524, 116726, 117021, 117122, 117324, 117930, 118023, 118124, 118225, 118528,
                      118730, 118831, 118932, 119025, 119126, 120111, 120212, 120414, 120515, 120717, 121416, 121618,
                      121921, 122317, 122620, 122822, 123117, 123420, 123521, 123723, 123824, 123925, 124220, 124422,
                      124624, 124826, 125222, 125424, 125525, 126325, 126426, 126628, 127226, 127327, 127630, 127731,
                      127832, 127933, 128026, 128127, 128632, 128935, 129028, 129129, 129331, 129634, 130013, 130114,
                      130316, 130417, 130518, 130619, 130720, 130821, 130922, 131217, 131419, 131722, 131823, 131924,
                      132017, 132118, 133019, 133625, 133827, 133928, 134021, 134223, 134324, 134425, 134627, 134728,
                      134829, 135124, 135225, 135528, 135629, 135730, 135932, 136126, 136227, 136631, 136732, 136833,
                      137027, 137128, 137229, 137431, 137532, 137633, 137936, 138130, 138231, 138332, 138534, 138837,
                      139233, 139435, 139637, 139839, 140117, 140319, 140824, 140925, 141119, 141422, 141826, 142828,
                      143224, 143325, 143426, 143830, 144125, 144226, 144428, 144731, 144832, 144933, 145127, 145632,
                      145834, 146129, 146331, 146432, 146533, 146735, 146836, 146937, 147030, 147636, 147737, 148032,
                      148133, 148335, 148436, 148840, 148941, 149236, 149337, 149539, 149741, 149842, 150625, 150726,
                      150928, 151223, 151324, 151425, 151526, 151627, 151728, 151829, 151930, 152225, 152427, 152831,
                      153025, 153126, 153227, 153429, 153631, 153732, 153833, 153934, 154229, 154330, 154431, 154532,
                      154734, 154835, 154936, 155635, 155938, 156031, 156233, 156334, 156435, 156536, 156637, 157336,
                      157437, 157942, 158035, 158136, 158338, 158540, 158843, 159138, 159239, 159340, 159441, 159744,
                      160123, 160729, 160830, 161327, 161630, 161731, 161832, 162026, 162228, 162329, 162733, 162935,
                      163129, 163331, 163432, 163836, 164030, 164131, 164636, 164939, 165032, 165436, 165638, 165840,
                      165941, 166438, 166640, 167036, 167238, 167440, 167743, 168139, 168240, 168341, 168745, 168947,
                      169040, 169343, 169444, 169545, 169949, 170631, 171330, 171532, 171633, 172029, 172130, 172332,
                      172433, 172534, 172938, 173334, 173435, 173536, 173637, 173738, 173839, 173940, 174437, 174841,
                      175035, 175136, 175237, 175338, 175439, 175540, 175742, 176037, 176239, 176441, 176542, 176744,
                      176845, 177140, 177241, 177645, 177746, 178142, 178243, 178647, 178748, 178849, 178950, 179245,
                      179346, 180129, 180230, 180432, 180533, 180735, 180836, 180937, 181131, 181232, 181636, 182032,
                      182436, 182739, 182840, 183034, 185038, 185139, 185341, 185442, 185846, 185947, 186040, 186141,
                      186444, 186545, 186848, 187143, 187345, 187547, 187850, 188145, 188347, 188448, 188549, 188751,
                      189349, 189450, 189652, 190031, 191033, 191235, 191336, 191437, 191841, 191942, 192035, 192136,
                      192237, 192439, 192540, 192641, 192843, 193239, 193845, 194140, 194443, 194645, 194746, 194847,
                      195041, 195445, 195647, 195849, 195950, 196144, 196346, 196750, 197348, 197550, 198047, 198249,
                      198350, 198451, 198653, 198855, 199150, 199251, 199352, 199453, 199655, 199958, 200008, 200109,
                      200311, 200513, 200614, 200917, 201111, 201414, 201515, 201818, 202113, 202719, 203418, 203923,
                      204016, 204218, 204319, 204420, 204521, 204622, 205119, 205220, 205725, 205826, 206222, 206323,
                      206525, 206727, 206828, 206929, 207123, 207426, 208024, 208125, 208226, 208327, 208630, 209127,
                      209228, 209329, 209834, 209935, 210011, 210112, 210415, 210617, 211114, 211215, 211316, 211417,
                      211619, 211720, 211821, 211922, 212015, 212116, 212217, 212318, 212419, 212823, 213017, 213421,
                      213522, 214019, 214221, 214423, 214524, 214625, 214726, 217126, 217429, 219231, 220721, 221319,
                      223929, 224022, 227432, 227533, 228434, 231928, 233326, 236130, 237334, 238033, 239136, 239944,
                      245333, 246133, 248339, 249947, 250427, 250932, 251833, 255639, 255740, 256540, 257542, 257845,
                      257946, 263436, 268749, 268850, 270332, 274542, 275645, 280739, 280941, 281135, 283543, 285345,
                      285446, 286347, 286650, 287248, 289555, 290136, 293748, 295146, 297655, 298051, 298455, 299154,
                      299760, 300618, 300719, 303119, 303624, 304020, 304727, 305830, 307127, 308129, 308331, 309636,
                      310621, 311320, 314225, 316633, 316835, 318637, 320826, 321323, 322224, 325129, 329440, 329844,
                      330324, 333330, 334635, 336841, 339847, 341834, 342129, 346137, 346945, 348545, 349244, 350330,
                      352132, 352738, 353740, 356948, 358144, 360030, 361234, 361941, 365343, 366042, 366446, 368551,
                      368753, 371843, 376247, 377451, 378756, 378857, 379657, 380036, 381038, 381543, 382242, 385046,
                      385450, 386250, 387959, 389357, 390645, 391748, 392447, 392750, 393247, 393550, 394956, 395251,
                      395756, 395958, 397154, 397760, 397861, 401422, 406432, 406836, 412528, 413934, 414229, 415837,
                      419239, 421226, 422632, 424939, 429040, 432332, 433839, 436239, 436845, 441939, 445543, 448347,
                      449753, 453441, 453542, 454140, 456346, 459453, 461743, 463040, 465852, 467351, 468050, 469961,
                      475855, 479762, 480141, 481042, 481951, 485757, 486759, 495255, 497865, 499566, 500222, 506234,
                      510225, 510326, 512835, 513130, 513736, 516742, 517239, 518746, 519647, 519950, 520228, 522434,
                      523032, 524135, 525541, 529549, 529953, 530635, 531536, 531940, 536647, 540436, 541640, 541943,
                      545345, 547046, 548250, 552241, 552544, 553344, 555348, 555651, 555954, 557857, 558657, 558960,
                      559053, 559457, 561242, 561444, 561949, 562345, 562446, 565452, 566454, 567052, 567759, 567961,
                      568963, 570243, 571144, 572045, 573249, 573451, 576255, 578057, 579665, 579867, 580044, 580347,
                      580650, 580751, 581349, 581450, 583858, 585256, 585862, 586460, 587664, 588565, 589567, 590047,
                      592455, 594156, 597869, 598568, 599065, 599469, 599671, 601127, 604537, 609143, 611938, 613538,
                      614439, 615441, 615744, 616645, 617748, 618952, 620434, 622236, 623844, 626648, 627549, 627852,
                      628248, 633847, 634748, 635245, 638049, 645450, 645551, 647858, 654350, 654552, 654754, 656253,
                      656657, 657659, 660951, 662551, 663755, 664757, 665254, 667056, 668361, 671855, 672756, 673455,
                      675661, 677766, 677968, 679568, 679770, 680250, 680452, 680957, 683256, 685058, 686969, 687163,
                      690152, 692964, 693764, 694362, 695768, 698168, 700634, 702133, 704238, 705341, 706040, 707749,
                      709551, 713239, 715041, 715647, 715950, 720337, 723141, 724446, 725751, 727553, 727654, 728454,
                      729254, 729557, 731140, 732243, 734045, 735148, 737960, 742549, 744553, 748258, 748662, 749058,
                      749361, 751348, 753150, 753251, 756055, 757764, 759869, 760551, 761957, 763557, 765056, 765864,
                      767464, 769064, 770352, 771354, 773257, 774663, 779370, 782561, 783462, 784565, 788674, 788876,
                      789373, 792564, 792766, 792867, 793465, 800941, 802844, 803240, 804646, 809252, 810843, 812746,
                      814548, 814649, 815247, 816653, 818455, 818859, 820745, 825048, 825553, 825654, 826353, 826454,
                      827052, 828862, 832651, 833148, 833249, 835657, 837560, 837964, 841349, 843151, 844961, 845458,
                      849264, 849971, 852455, 856766, 856968, 857263, 859671, 861456, 865363, 867468, 869472, 870861,
                      871762, 871964, 872158, 872562, 872764, 873968, 877168, 877269, 878776, 878877, 880157, 882161,
                      884064, 885975, 886674, 887373, 888678, 889579, 891667, 894067, 894673, 894774, 896778, 896879,
                      898176, 899885, 901038, 901139, 901442, 902242, 904044, 905147, 907656, 908860, 910241, 910443,
                      911849, 912447, 917255, 917558, 919966, 922854, 923755, 926862, 927359, 930449, 932554, 933253,
                      937160, 942658, 943862, 947668, 951457, 952863, 955465, 957974, 958976, 959574, 962058, 965367,
                      965771, 966975, 969476, 970764, 971160, 978578, 979984, 983773, 984472, 987074, 987983, 989987,
                      990366, 991267, 992673, 992774, 993675, 994273, 996782]


def get_timeseries_final_path(person, session_day, direction=False):
    if not direction:
        return f'../hcp_multimodal_parcellation/concatenated_timeseries/{person}_{session_day}.npy'
    else:
        return (f'../hcp_multimodal_parcellation/concatenated_timeseries/{person}_{session_day}_LR.npy',
                f'../hcp_multimodal_parcellation/concatenated_timeseries/{person}_{session_day}_RL.npy')

def merge_y_and_others(ys, indices):
    tmp = torch.cat([ys.long().view(-1, 1),
                     indices.view(-1, 1)], dim=1)
    return LabelEncoder().fit_transform([str(l) for l in tmp.numpy()])


def create_name_for_hcp_dataset(num_nodes, time_length, target_var, threshold, connectivity_type, normalisation,
                                disconnect_nodes=False,
                                prefix_location='./pytorch_data/balanced_hcp_4split_'):
    if time_length == 75:
        prefix_location = './pytorch_data/balanced_hcp_64split_'
    name_combination = '_'.join(
        [target_var, connectivity_type.value, str(num_nodes), str(threshold), normalisation.value,
         str(disconnect_nodes)])

    return prefix_location + name_combination

def create_best_encoder_name(ts_length, outer_split_num, encoder_name,
                             prefix_location = 'logs/',
                             suffix = '.pth'):
        return f'{prefix_location}{encoder_name}_{ts_length}_{outer_split_num}_best{suffix}'

def create_name_for_encoder_model(ts_length, outer_split_num, encoder_name,
                                  params,
                                  prefix_location='logs/',
                                  suffix='.pth'):
    return prefix_location + '_'.join([encoder_name,
                                       str(ts_length),
                                       str(outer_split_num),
                                       str(params['weight_decay']),
                                       str(params['lr'])
                                       ]) + suffix


def create_name_for_model(target_var, model, params, outer_split_num, inner_split_num, n_epochs, threshold, batch_size,
                          remove_disconnect_nodes, num_nodes, conn_type, normalisation,
                          analysis_type, metric_evaluated,
                          prefix_location='logs/',
                          suffix='.pth'):
    if analysis_type == AnalysisType.SPATIOTEMOPRAL:
        model_str_representation = model.to_string_name()
    elif analysis_type == AnalysisType.FLATTEN_CORRS or analysis_type == AnalysisType.FLATTEN_CORRS_THRESHOLD:
        suffix = '.pkl'
        model_str_representation = analysis_type.value
        for key in ['min_child_weight', 'gamma', 'subsample', 'colsample_bytree', 'max_depth', 'n_estimators']:
            model_str_representation += key + '_' + str(model.get_xgb_params()[key])
        params['lr'] = None
        params['weight_decay'] = None

    return prefix_location + '_'.join([target_var,
                                       str(outer_split_num),
                                       str(inner_split_num),
                                       metric_evaluated,
                                       model_str_representation,
                                       str(params['lr']),
                                       str(params['weight_decay']),
                                       str(n_epochs),
                                       str(threshold),
                                       normalisation.value,
                                       str(batch_size),
                                       str(remove_disconnect_nodes),
                                       str(num_nodes),
                                       conn_type.value
                                       ]) + suffix


# From https://www.kaggle.com/jakubwasikowski/stratified-group-k-fold-cross-validation
class StratifiedGroupKFold():

    def __init__(self, n_splits=5, random_state=0):
        self.n_splits = n_splits
        self.random_state = random_state

    def split(self, X, y, groups):
        labels_num = np.max(y) + 1
        y_counts_per_group = defaultdict(lambda: np.zeros(labels_num))
        y_distr = Counter()
        for label, g in zip(y, groups):
            y_counts_per_group[g][label] += 1
            y_distr[label] += 1

        y_counts_per_fold = defaultdict(lambda: np.zeros(labels_num))
        groups_per_fold = defaultdict(set)

        def eval_y_counts_per_fold(y_counts, fold):
            y_counts_per_fold[fold] += y_counts
            std_per_label = []
            for label in range(labels_num):
                label_std = np.std([y_counts_per_fold[i][label] / y_distr[label] for i in range(self.n_splits)])
                std_per_label.append(label_std)
            y_counts_per_fold[fold] -= y_counts
            return np.mean(std_per_label)

        groups_and_y_counts = list(y_counts_per_group.items())
        random.Random(self.random_state).shuffle(groups_and_y_counts)

        for g, y_counts in sorted(groups_and_y_counts, key=lambda x: -np.std(x[1])):
            best_fold = None
            min_eval = None
            for i in range(self.n_splits):
                fold_eval = eval_y_counts_per_fold(y_counts, i)
                if min_eval is None or fold_eval < min_eval:
                    min_eval = fold_eval
                    best_fold = i
            y_counts_per_fold[best_fold] += y_counts
            groups_per_fold[best_fold].add(g)

        all_groups = set(groups)
        for i in range(self.n_splits):
            train_groups = all_groups - groups_per_fold[i]
            test_groups = groups_per_fold[i]

            train_indices = [i for i, g in enumerate(groups) if g in train_groups]
            test_indices = [i for i, g in enumerate(groups) if g in test_groups]

            yield train_indices, test_indices

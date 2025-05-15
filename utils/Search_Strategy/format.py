import re
from typing import Union, List


def find_indices(text: str, return_item: str = 'iter') -> Union[List[int], re.Match]:
    # Define a regular expression to match numbers at the start, end, or surrounded by spaces, or between / and -
    pattern = r'(?:(?<=\s)|(?<=^)|(?<=/)|(?<=-)|(?<=\()|(?<=\sor\s))\d+(?:(?=\s)|(?=$)|(?=/)|(?=-)|(?=\))|(?=\sor\s))' 

    # Find all matching index numbers
    matches = re.finditer(pattern, text)
    if return_item == 'iter':
        return matches

    # Return all found index numbers
    return [int(match.group()) for match in matches]


def modify_indices(text: str, difference_value: int) -> str:
    # Find all index numbers
    matches = find_indices(text)

    # Create an offset to prevent errors caused by index changes during replacement
    offset = 0

    # Add the difference value to the found index numbers
    for match in matches:
        original_index = int(match.group())
        new_index = original_index + difference_value
        start, end = match.start() + offset, match.end() + offset

        # Replace the old number with the new number
        text = text[:start] + str(new_index) + text[end:]

        # Update the offset
        offset += len(str(new_index)) - len(match.group())

    return text


def add_filters(search_strategy: str, filters: Union[str, List[list]]) -> str:
    if isinstance(filters, str):
        filters = [['or', filters]]

    main_ss_last_index = int(search_strategy.split('\n')[-1].split()[0])
    last_index = 0
    last_index_map = []
    for f_relation, f_value in filters:
        first_index = int(f_value.split('\n')[0].split()[0])  # The first index of the current filter
        difference_value = (
            main_ss_last_index - first_index + 1
            if last_index == 0
            else last_index - first_index + 1
        )  # Calculate the first index of the current filter in the main search strategy, including previous filters
        new_f = modify_indices(f_value, difference_value)
        tmp_index = int(new_f.split('\n')[-1].split()[0])  # The last index of the current filter
        if main_ss_last_index not in find_indices(new_f, return_item='list'):
            and_str = f'{tmp_index+1}	{main_ss_last_index} and {tmp_index}'
            search_strategy += '\n' + new_f + '\n' + and_str
            last_index = tmp_index + 1
        else:
            search_strategy += '\n' + new_f
            last_index = tmp_index
        last_index_map.append((f_relation, last_index))

    if len(last_index_map) > 1:
        print(last_index_map)
        for r, _ in last_index_map:
            if r == 'and':
                raise NotImplementedError('The "and" case is not supported yet')

        merge_str = f'{last_index+1}	{last_index_map[0][1]} {last_index_map[0][0]} {last_index_map[1][1]}'
        for i in range(2, len(last_index_map)):
            merge_str += f' {last_index_map[i-1][0]} {last_index_map[i][1]}'
        search_strategy += '\n' + merge_str

    return search_strategy


def correct_format(search_strategy: str):
    search_strategy_list = search_strategy.split('\n')
    reformatted_search_strategy_list = [
        ' '.join(s.split(' ')[1:]) for s in search_strategy_list
    ]
    for i in range(len(reformatted_search_strategy_list)):
        idx = i + 1
        reformatted_search_strategy_list[i] = (
            str(idx) + '     ' + reformatted_search_strategy_list[i]
        )
    reformatted_search_strategy = '\n'.join(reformatted_search_strategy_list)
    return reformatted_search_strategy


if __name__ == '__main__':

    input_text = '''64     limit 63 to in process
65     (MEDLINE or systematic review).tw. or meta analysis.pt. [McMaster University Health Research
            Information Unit. Filter for identifying reviews - best balance of specificity and sensitivity.
            https://hiru.mcmaster.ca/hiru/HIRU_Hedges_MEDLINE_Strategies.aspx]
66     63 and 65
67     limit 66 to english language [drug pico and systematic reviews and English]
68     (randomized controlled trial or controlled clinical trial).pt. or (randomized or placebo).ab. or drug
            therapy.fs. or (randomly or trial or groups).ab. [Cochrane Highly Sensitive Search
            Strategy for identifying randomized trials in Ovid Medline]
69     epidemiologic studies/ or exp case control studies/ or exp cohort studies/ or Cross-sectional studies/
70     case control.tw.
71     (cohort adj (study or studies)).tw.
72     cohort analy$.tw.
73     (Follow up adj (study or studies)).tw.
74     (Observational adj (study or studies)).tw.
75     (longitudinal or retrospective or cross sectional).tw.
76     or/69-75 [Scottish Intercollegiate Guidelines SIGN https://www.sign.ac.uk/search-filters.html]
77     68 or 76
78     63 and 77 [drug picos and rct, observ]
79     exp adult/ or aged.sh. or age:.tw. or adult.mp. or middle aged.sh. or young adult.sh. or geriatric.tw.
80     78 and 79
81     limit 80 to english language [drug picos and rct,  observ and age groups and english]
    '''
    # input_text = "22\trandomized controlled trial.pt.\n23\tcontrolled clinical trial.pt.\n24\trandomized.ab.\n25\tplacebo.ab.\n26\tclinical trials as topic.sh.\n27\trandomly.ab.\n28\ttrial.ti.\n29\t22 or 23 or 24 or 25 or 26 or 27 or 28\n30\tEpidemiologic studies/\n31\texp case control studies/\n32\texp cohort studies/\n33\tCase control.tw.\n34\t(cohort adj (study or studies)).tw.\n35\tCohort analy$.tw.\n36\t(Follow up adj (study or studies)).tw.\n37\t(observational adj (study or studies)).tw.\n38\tLongitudinal.tw.\n39\tRetrospective.tw.\n40\tCross sectional.tw.\n41\tCross-sectional studies/\n42\t30 or 31 or 32 or 33 or 34 or 35 or 36 or 37 or 38 or 39 or 40 or 41\n43\t29 or 42"

    filter1 = '''64     limit 63 to in process'''
    filter2 = '''65     (MEDLINE or systematic review).tw. or meta analysis.pt. [McMaster University Health Research
            Information Unit. Filter for identifying reviews - best balance of specificity and sensitivity.
            https://hiru.mcmaster.ca/hiru/HIRU_Hedges_MEDLINE_Strategies.aspx]
66     63 and 65
67     limit 66 to english language [drug pico and systematic reviews and English]'''
    filter3 = '''68     (randomized controlled trial or controlled clinical trial).pt. or (randomized or placebo).ab. or drug
            therapy.fs. or (randomly or trial or groups).ab. [Cochrane Highly Sensitive Search
            Strategy for identifying randomized trials in Ovid Medline]
69     epidemiologic studies/ or exp case control studies/ or exp cohort studies/ or Cross-sectional studies/
70     case control.tw.
71     (cohort adj (study or studies)).tw.
72     cohort analy$.tw.
73     (Follow up adj (study or studies)).tw.
74     (Observational adj (study or studies)).tw.
75     (longitudinal or retrospective or cross sectional).tw.
76     or/69-75 [Scottish Intercollegiate Guidelines SIGN https://www.sign.ac.uk/search-filters.html]
77     68 or 76
78     63 and 77 [drug picos and rct, observ]
79     exp adult/ or aged.sh. or age:.tw. or adult.mp. or middle aged.sh. or young adult.sh. or geriatric.tw.
80     78 and 79
81     limit 80 to english language [drug picos and rct,  observ and age groups and english]'''
    input_text = [['or', filter1], ['or', filter2], ['or', filter3]]


    search_strategy = '''1     Arthritis, Rheumatoid/ 
2     ((rheumatoid adj2 arthrit$) or (rheumatoid adj2 arthros$)).tw,kw. 
3     Rheumatoid nodule/ or (rheumatoid adj2 nodule$).tw,kw. 
4     or/1-3 
5     Hydroxychloroquine/ or (hydroxychloroquine or Plaquenil).tw,kw. 
6     (leflunomide or Arava).tw,kw. 
7     Methotrexate/ or (methotrexate or Rheumatrex or Trexall or Otrexup or Rasuvo).tw,kw. 
8     Sulfasalazine/ or (sulfasalazine$ or sulphasalazine$ or Azulfidine).tw,kw. 
9     or/5-8 [conventional synthetic DMARDs] 
10     Adalimumab/ or (adalimumab or Humira).tw,kw. 
11     Certolizumab Pegol/ or (certolizumab or Cimzia).tw,kw. 
12     Etanercept/ or (etanercept or Enbrel).tw,kw. 
13     (Golimumab or Simponi).tw,kw. 
14     Infliximab/ or (infliximab or Remicade).tw,kw. 
15     Tumor Necrosis Factor-alpha/tu [Therapeutic Use] 
16     Receptors, Tumor Necrosis Factor/tu 
17     ((tumor or tumour) adj necrosis factor adj (block$ or inhibitor$)).tw. or (TNF inhibitor$ or TNFi).tw,kw. or (anti-TNF or anti TNF).tw,kw. 
18     Abatacept/ or (abatacept or Orencia).tw,kw. 
19     Rituximab/ or (rituximab or Rituxan or MabThera or Remsima).tw,kw. 
20     Antibodies, Monoclonal/tu 
21     (Sarilumab or Kevzara).tw,kw. 
22     tocilizumab.tw,kw. 
23     Interleukin-6/tu or Interleukin-1/tu or Receptors, Interleukin-6/tu or Receptors, Interleukin-1/tu 
24     (interleukin 6 inhibit$ or IL 6 inhibit$).tw,kw. 
25     biologic$ DMARD$.tw,kw. 
26     biological products/ and exp antirheumatic agents/ 
27     or/10-26 [biological DMARDs] 
28     (Amjevita or "adalimumab-atto").tw,kw. 
29     (Erelzi or "etanercept szzs").tw,kw.
30     (infliximab-QBTZ or Ixifi).tw,kw. 
31     ("Infliximab-DYBB" or Inflectra).tw,kw. 
32     biosimilar pharmaceuticals/ and antirheumatic agents/ 
33     (biosimilar adj2 disease adj2 modifying adj2 anti-rheumatic$).tw,kw. 
34     (biosimilar adj3 DMARD$).tw,kw. 
35     (bs adj DMARD$).tw,kw. 
36     or/28-35 [biosimiar dmards] 
37     (baricitinib or Olumiant).tw,kw. 
38     (fostamatanib or Tavalisse).tw,kw. 
39     (Filgotinib or GS-6034 or GLPG0634).tw,kw. 
40     (tofacitinib or tasocitinib or Xeljanz or cp690550 or "cp 690550" or "cp 690 550" or Upadacitinib or “ABT-494”).tw,kw. 
41     (((JAK 1 or JAK 2) adj inhibit$) or (Janus kinase 1 inhibit$ or Janus kinase 3 inhibit$)).tw,kw. 
42     Janus Kinase inhibitors/ or Janus Kinase 1/ai or Janus Kinase 3/ai 
43     or/37-42 [targeted dmards]
44     4 and (9 or 27 or 36 or 43) [PICO 21-23,55]'''

    final_search_strategy = add_filters(search_strategy, input_text)
    print(final_search_strategy)
    # print(input_text)

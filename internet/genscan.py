import requests
import re
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup


class GenscanOutput():
    def __init__(self, status, cds_list, intron_list, exon_list):
        self.status = status
        self.cds_list = cds_list
        self.intron_list = intron_list
        self.exon_list = exon_list


def run_genscan(sequence=None, sequence_file=None, organism="Vertebrate", exon_cutoff=1.00, sequence_name=""):
    url = 'http://hollywood.mit.edu/cgi-bin/genscanw_py.cgi'
    form_data = {'-o': organism,
                 '-e': exon_cutoff,
                 '-n': sequence_name,
                 '-p': 'Predicted peptides only',
                 '-s': sequence}
    form_files = None
    if sequence_file:
        file = open(sequence_file, 'rb')
        form_files = {'-u': file}

    response = requests.post(url, data=form_data, files=form_files)
    status = response.status_code

    soup = BeautifulSoup(response.content, "lxml")
    element = soup.find("pre").text

    template = r"\d+_aa([\n\w]+)"
    match_list = re.findall(template, str(element))

    if len(match_list) == 0:
        output = GenscanOutput(status, [], [], [])
    else:
        cds_list = [peptide.replace('\n', '') for peptide in match_list]

        number_peptides = len(cds_list)
        intron_list = []
        exon_list = []
        for n_peptide in range(1, number_peptides + 1):
            n_peptide = str(n_peptide)
            template = n_peptide + '\.\d\d (\w+) [+-] +(\d+) +(\d+)'
            match_list = re.findall(template, str(element))
            for i in range(len(match_list)):
                match_list[i] = list(match_list[i])
            match_df = pd.DataFrame(np.array(match_list).T, index=[0, 'start', 'stop'])
            match_df.columns = match_df.iloc[0]
            match_df.drop(match_df.index[0], inplace=True)
            match_df = match_df.astype(int)
            n_introns = match_df.shape[1] - 1

            if n_introns != 0:
                intron_dict = {}
                if match_df.columns[0] in ['Prom', 'Init']:
                    for n in range(n_introns):
                        intron_dict[f'intron_{n + 1}'] = [match_df.iloc[1, n] + 1, match_df.iloc[0, n + 1] - 1]
                else:
                    for n in range(n_introns):
                        intron_dict[f'intron_{n + 1}'] = [match_df.iloc[0, n] + 1, match_df.iloc[1, n + 1] - 1]

                introns = pd.DataFrame(intron_dict, index=['start', 'stop'])
            else:
                introns = None

            exon_names = ['Init', 'Intr', 'Term', 'Sngl']
            column_names = match_df.columns
            exon_columns = list(set(exon_names) & set(column_names))
            exons = match_df.loc[:, exon_columns]

            intron_list.append(('Peptide_' + n_peptide, introns))
            exon_list.append(('Peptide_' + n_peptide, exons))

            output = GenscanOutput(status, cds_list, intron_list, exon_list)
    return output
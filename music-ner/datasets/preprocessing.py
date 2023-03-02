import re
import nltk
import pandas as pd
import argparse


class WrittenQueryProcessor:
    def __init__(self):
        self.keep = {'$', '&', '+', '@', '¿'}
        self.discard = {'"', "'", '*', '«', '»', '́', '‘', '’', '“', '”', '„'}
        self.punctmarks = {'.', '?', '!'}
        self.start_parantheses = {'(', '['}
        self.end_parantheses = {')', ']'}
        self.newline = {';', '|'}
        self.ignore_2char_words = {'st', 'he', 'if', 'do', 'in', 'is', 'it', 'me', 'up', 'so', 'to', 'us'}
        self.special_abbrv = {'remix', 'prod', 'vol', 'mvt', 'mix', 'feat', 'alt', 'aka'}
        self.MIN_SENT = 3

    def processing_pipeline(self, sents):
        result = []
        for sent in sents:
            sent = sent.lower()

            # Step 1: the basic processing neglects the more complicated cases and treats only the cases where the cleaning is deterministic (a non-alphanumeric character of a certain type is replaced either with space, empty string or newline)
            sent = self.process_sent_basic(sent)

            # Step 2: deal with punctuation marks when they repeat
            sent = self.remove_punctmark_repetitions(sent)

            # Step 3: deal with punctuation marks when they are at the end
            sent = self.remove_final_punctmark(sent)

            # Step 4: remove the punctuation marks inside the words
            sent = self.remove_punctmark_inword(sent)

            # Step 5: treat special cases involving dot
            sent = self.remove_dot_special_cases(sent)

            # Step 6: split by punctuation marks and remove the marks
            sent = self.process_final_punctmark(sent)

            # Step 7: deal with the parantheses; most of the cases will create newlines apart from the case when the parantheses are inside or are part of emoticons, being replaced then by empty string
            sent = self.process_parantheses(sent)

            result.append(sent)
        return result

    def process_sent_basic(self, sent):
        original_characters = list(sent)
        process_characters = []
        for c in original_characters:
            if c.isalnum() or c in self.keep or c in self.start_parantheses or c in self.end_parantheses or c in self.punctmarks:
                process_characters.append(c)
            else:
                if c in self.discard:
                    continue
                elif c in self.newline:
                    process_characters.append('\n')
                else:
                    process_characters.append(' ')
        return ''.join(process_characters).strip()

    def remove_punctmark_repetitions(self, sent):
        return re.sub(r'[\.|?|!]{2,}', ' ', sent)

    def remove_final_punctmark(self, sent):
        if sent[-1] in self.punctmarks:
            return sent[:-1]
        return sent

    def remove_punctmark_inword(self, sent):
        chs = []
        for i in range(len(sent)):
            if not (sent[i] in self.punctmarks and i > 0 and sent[i - 1] != ' ' and i < len(sent) - 1 and sent[i + 1] != ' '):
                chs.append(sent[i])
        return ''.join(chs)

    def remove_dot_special_cases(self, text):
        sents = text.split('\n')
        final_sents = []
        for sent in sents:
            words = sent.split()
            for i in range(len(words)):
                if len(words[i]) < 2 or len(words[i]) > 6 or '.' not in words[i]:
                    continue
                if words[i][0] == '.':
                    words[i] = words[i].replace('.', '')
                elif words[i][-1] == '.':
                    # Cases covered single or 2 letters followed by dot except when these cases are numbers of known vocabulary words
                    if len(words[i][:-1]) <= 2 and words[i][:-1] not in self.ignore_2char_words and not words[i][:-1].isnumeric():
                        words[i] = words[i].replace('.', '')
                    elif words[i][:-1] in self.special_abbrv:
                        words[i] = words[i].replace('.', '')
            final_sents.append(' '.join(words))
        return '\n'.join(final_sents).strip()

    def process_final_punctmark(self, text):
        sents = text.split('\n')
        final_sents = []
        for sent in sents:
            nltk_sents = nltk.sent_tokenize(sent)
            final_sent = '\n'.join(nltk_sents)
            for p in self.punctmarks:
                final_sent = re.sub(r'[\.|?|!]', '', final_sent)
            final_sent = final_sent.strip()
            if len(final_sent) > self.MIN_SENT:
                final_sents.append(final_sent)
        return '\n'.join(final_sents).strip()

    def process_parantheses(self, text):
        sents = text.split('\n')
        final_sents = []
        for i in range(len(sents)):
            if sents[i] is None or sents[i] == '':
                continue
            sents[i] = self.process_start_parantheses(sents[i])
            sents[i] = self.process_end_parantheses(sents[i])
            sents[i] = self.process_rest_of_parantheses(sents[i])
            sents[i] = sents[i].strip()
            if len(sents[i]) > self.MIN_SENT:
                final_sents.append(sents[i])
        return '\n'.join(final_sents)

    def process_start_parantheses(self, sent):
        sent = sent.strip()
        if sent == '':
            return sent
        if sent[0] in self.start_parantheses:
            if sent[0] == '(':
                if ')' in sent:
                    index = sent.index(')')
                else:
                    if len(sent) < self.MIN_SENT - 1:
                        return ''
                    return sent
            else:
                if ']' in sent:
                    index = sent.index(']')
                else:
                    if len(sent) < self.MIN_SENT - 1:
                        return ''
                    return sent
            head = sent[1:index].strip()
            if len(head) < self.MIN_SENT - 1:
                return self.process_start_parantheses(sent[index + 1:].strip())
            else:
                return head + '\n' + self.process_start_parantheses(sent[index + 1:].strip())
        if len(sent) < self.MIN_SENT - 1:
            return ''
        return sent

    def process_end_parantheses(self, sent):
        sent = sent.strip()
        if sent == '':
            return sent
        if sent[-1] in self.end_parantheses:
            if sent[-1] == ')':
                if '(' in sent:
                    index = sent.rindex('(')
                else:
                    if len(sent) < self.MIN_SENT - 1:
                        return ''
                    return sent
            else:
                if '[' in sent:
                    index = sent.rindex('[')
                else:
                    if len(sent) < self.MIN_SENT - 1:
                        return ''
                    return sent
            tail = sent[index + 1:-1].strip()
            if len(tail) < self.MIN_SENT - 1:
                return self.process_end_parantheses(sent[:index].strip())
            else:
                return self.process_end_parantheses(sent[:index].strip()) + '\n' + tail
        if len(sent) < self.MIN_SENT - 1:
            return ''
        return sent

    def process_rest_of_parantheses(self, sent):
        if '(' in sent or ')' in sent or '[' in sent or ']' in sent:
            original_characters = list(sent)
            process_characters = []
            for c in original_characters:
                if c not in self.start_parantheses and c not in self.end_parantheses:
                    process_characters.append(c)
            return ''.join(process_characters)
        return sent


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--original_corpus",
        dest="original_corpus",
        type=str,
        help="Original corpus path",
        required=True,
    )
    parser.add_argument(
        "--output_csv",
        dest="output_csv",
        type=str,
        help="Output csv filepath",
        required=True,
    )
    args = parser.parse_args()

    processor = WrittenQueryProcessor()
    with open(args.original_corpus, 'r') as _:
        sents = [s.replace('\n', '') for s in _.readlines()]
        results = processor.processing_pipeline(sents)
        header = ['preprocessed', 'original']
        df = pd.DataFrame(zip(results, sents), columns=header)
        df.to_csv(args.output_file, index=False)

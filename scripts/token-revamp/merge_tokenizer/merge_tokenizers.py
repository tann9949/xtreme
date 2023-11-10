from argparse import ArgumentParser, Namespace
import os

from transformers import CamembertTokenizer, XLMRobertaTokenizer
from sentencepiece import sentencepiece_model_pb2 as sp_pb2_model
import sentencepiece as spm

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"]="python"


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument('--xlmr_tokenizer_dir', default="xlm-roberta-base", type=str)
    parser.add_argument('--wangchanberta_sp_model_file', default='./wangchanberta_sp.model', type=str)
    args = parser.parse_args()
    return args

def main(args: Namespace) -> None:
    xlmr_tokenizer_dir = args.xlmr_tokenizer_dir
    wangchanberta_sp_model_file = args.wangchanberta_sp_model_file

    # load tokenizers
    xlmr_tokenizer = XLMRobertaTokenizer.from_pretrained(xlmr_tokenizer_dir)
    wangchanberta_sp_model = spm.SentencePieceProcessor()
    wangchanberta_sp_model.Load(wangchanberta_sp_model_file)

    xlmr_spm = sp_pb2_model.ModelProto()
    xlmr_spm.ParseFromString(xlmr_tokenizer.sp_model.serialized_model_proto())
    wangchanberta_spm = sp_pb2_model.ModelProto()
    wangchanberta_spm.ParseFromString(wangchanberta_sp_model.serialized_model_proto())

    # print number of tokens
    print(len(xlmr_tokenizer), len(wangchanberta_sp_model))
    print(xlmr_tokenizer.all_special_tokens)
    print(xlmr_tokenizer.all_special_ids)
    print(xlmr_tokenizer.special_tokens_map)

    ## Add Thai tokens to XLMR tokenizer
    xlmr_spm_tokens_set = set(p.piece for p in xlmr_spm.pieces)
    print(len(xlmr_spm_tokens_set))
    print(f"Before:{len(xlmr_spm_tokens_set)}")
    num_overlap_vocab = 0
    num_new_vocab = 0
    for p in wangchanberta_spm.pieces:
        piece = p.piece
        if piece not in xlmr_spm_tokens_set:
            new_p = sp_pb2_model.ModelProto().SentencePiece()
            new_p.piece = piece
            new_p.score = 0
            xlmr_spm.pieces.append(new_p)
            num_new_vocab += 1
        else:
            num_overlap_vocab += 1
    print(f"New model pieces: {len(xlmr_spm.pieces)}")
    print(f"# Overlap vocab: {num_overlap_vocab}")
    print(f"# New vocab: {num_new_vocab}")

    ## Save
    output_sp_dir = 'merged_tokenizer_sp'
    output_hf_dir = 'merged_tokenizer_hf' # the path to save WangchanXLM-RoBERTa tokenizer
    os.makedirs(output_sp_dir, exist_ok=True)
    os.makedirs(output_hf_dir, exist_ok=True)
    with open(output_sp_dir+'/wangchanx_roberta.model', 'wb') as f:
        f.write(xlmr_spm.SerializeToString())
    tokenizer = XLMRobertaTokenizer(vocab_file=output_sp_dir+'/wangchanx_roberta.model')

    tokenizer.save_pretrained(output_hf_dir)
    print(f"WangchanXLM-RoBERTa tokenizer has been saved to {output_hf_dir}")


    # Test
    xlmr_tokenizer = XLMRobertaTokenizer.from_pretrained(xlmr_tokenizer_dir)
    wangchanxlmr_tokenizer = XLMRobertaTokenizer.from_pretrained(output_hf_dir)
    print(tokenizer.all_special_tokens)
    print(tokenizer.all_special_ids)
    print(tokenizer.special_tokens_map)
    text='''ทดสอบประโยคภาษาไทยที่มีการเปลี่ยนคำศัพท์จากโมเดลเอ็กซ์แอลเอ็มอาร์ให้รวมกับวังจันทร์เบอร์ต้า
    The primary use of WangchanXLMR is research on large language models, including'''
    print("Test text:\n",text)
    print(f"Tokenized by XLMR tokenizer:{xlmr_tokenizer.tokenize(text)}")
    print(f"Tokenized by WangchanXLM-R tokenizer:{wangchanxlmr_tokenizer.tokenize(text)}")


if __name__ == "__main__":
    args = parse_args()
    main(args)

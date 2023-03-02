#-*- coding: utf-8 -*-
# Implemented by Yoonseok Heo at 230110
# Email: nlp.ysheo419@gmail.com
# Code Reference URL
#  - VisualGPT: https://github.com/Vision-CAIR/VisualGPT
import os, json
#local_rank = int(os.environ["LOCAL_RANK"])
#os.environ["CUDA_VISIBLE_DEVICES"]="1"
import argparse, torch, itertools, evaluation, logging, random
import os.path as osp
import numpy as np
from ProcessData.myDataLoader import myDataset
from ProcessData.myDataCollator import myDataCollatorForSeq2Seq
from myModels.video_encoder import VideoEncoder
from myModels.gpt2_decoder import MyGPT2Decoder
from transformers import AdamW, GPT2Tokenizer
from torch.nn import NLLLoss
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from shutil import copyfile
from setproctitle import setproctitle


def get_parser():
    parser = argparse.ArgumentParser()

    # required
    parser.add_argument("--dataset_name", type=str, required=True, choices=["MSR-VTT", "MSVD"])
    parser.add_argument("--model_name", type=str, required=True,
                        choices=["gpt2", "gpt2-medium"])

    # Optional
    parser.add_argument("--debug_flag", action="store_true")
    parser.add_argument("--proctitle_name", type=str, default="train_py_debug")
    parser.add_argument("--logs_folder", type=str, default="tensorboard_logs")
    parser.add_argument("--exp_name", type=str, default="train_py_debug")
    parser.add_argument("--log_file", type=str, default="log/train_py_debug.txt")

    parser.add_argument("--data_dir_path", type=str, default="data")
    parser.add_argument("--feature_type", type=str, default="CLIP")
    #parser.add_argument("--output_directory_name", type=str, default="CLIP/features")
    parser.add_argument("--clip_pretrained_path", type=str, default="ViT-B/32")
    #parser.add_argument("--filter_threshold", type=float, default=0.9)
    #parser.add_argument("--max_sequence_frames", type=int, default=2000)

    parser.add_argument("--num_target_captions", type=int, default=10)

    # generation_config
    parser.add_argument("--max_sequence_length", type=int, default=25)      # also used in training
    parser.add_argument("--gen_sequence_length", type=int, default=40)      # only used in generation
    parser.add_argument("--num_beams", type=int, default=3)
    parser.add_argument("--gen_early_stopping", action="store_true")

    # modeling
    parser.add_argument("--max_frames", type=int, default=25)
    parser.add_argument('--tfm_heads', type=int, default=8)
    parser.add_argument('--tfm_layers', type=int, default=3)

    # p-tuning args
    parser.add_argument("--use_lm_finetune", type=bool, default=False)
    parser.add_argument("--pseudo_token", type=str, default="[PROMPT]")
    parser.add_argument("--template", type=str, default="(3, 3, 0)")

    # training args
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument("--reinforcement_lr", type=float, default=1e-5)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument('--max_grad_norm', default=1.0, type=float)
    parser.add_argument("--long_caption_flag", action="store_true", help="concat target captions into one sentence")
    parser.add_argument("--num_caption_concat", type=int, default=5)

    return parser


def train_xe(model, dataloader,gpt_optimizer,dataloader_eval,args, device, loss_fn, tokenizer, cur_epoch):
    # Training with cross-entropy
    model.train()
    running_loss = .0

    with tqdm(desc='Epoch %d - train' % cur_epoch, unit='it', total=len(dataloader)) as pbar:
        #for it, (padded_video_feats, attention_mask, target_captions, _, _) in enumerate(dataloader):
        for it, batch in enumerate(dataloader):
            padded_video_feats = batch.padded_video_feats.to(device)
            attention_mask = batch.attention_mask.to(device)
            target_captions = batch.caption_features.to(device)

            # out: batch, seq_len, vocab;  past: #blocks, 2, batch, nh, seq_len, feat/nh
            #out, past = model(padded_video_feats, attention_mask, target_captions)
            model_output = model(padded_video_feats, attention_mask, target_captions)
            lm_logits = model_output.logits
            past = model_output.past_key_values

            captions_gt = target_captions["input_ids"][:, 1:].contiguous()
            out = lm_logits[:, :-1].contiguous()

            loss = loss_fn(out.view(-1, tokenizer.vocab_size), captions_gt.view(-1))

            loss.backward()


            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            gpt_optimizer.step()
            gpt_optimizer.zero_grad()


            this_loss = loss.item()
            running_loss += this_loss

            pbar.set_postfix(loss=running_loss / (it + 1))
            pbar.update()

            # Debug
            # if it == 40:
            #     break

    loss = running_loss / len(dataloader)
    return loss

def evaluate_loss(model, dataloader, loss_fn, cur_epoch, device, tokenizer):
    model.eval()
    running_loss = 0.0
    with tqdm(desc='Epoch %d - validation' % cur_epoch, unit='it', total=len(dataloader)) as pbar:
        with torch.no_grad():
            #for it, (padded_video_feats, attention_mask, target_captions, all_caption_list, video_id_list) in enumerate(dataloader):
            for it, batch in enumerate(dataloader):
                padded_video_feats = batch.padded_video_feats.to(device)
                attention_mask = batch.attention_mask.to(device)
                target_captions = batch.caption_features.to(device)


                padded_video_feats = padded_video_feats.to(device)
                attention_mask = attention_mask.to(device)
                target_captions = target_captions.to(device)

                #out, past = model(padded_video_feats, attention_mask, target_captions)
                model_output = model(padded_video_feats, attention_mask, target_captions)
                lm_logits = model_output.logits
                past = model_output.past_key_values

                captions_gt = target_captions["input_ids"][:, 1:].contiguous()
                out = lm_logits[:, :-1].contiguous()

                loss = loss_fn(out.view(-1, tokenizer.vocab_size), captions_gt.view(-1))

                this_loss = loss.item()
                running_loss += this_loss

                pbar.set_postfix(loss=running_loss / (it + 1))
                pbar.update()
                #break   # debug

    val_loss = running_loss / len(dataloader)
    return val_loss

def evaluate_metrics(model, dataloader, tokenizer, exp_name, epoch, device, **gen_kwargs):
    model.eval()

    gen = {}
    gts = {}
    test_result = []
    with tqdm(desc='Epoch %d - evaluation' % epoch, unit='it', total=len(dataloader)) as pbar:
        #for it, (padded_video_feats, attention_mask, target_captions, all_caption_list, video_id_list) in enumerate(dataloader):
        for it, batch in enumerate(dataloader):
            padded_video_feats = batch.padded_video_feats.to(device)
            attention_mask = batch.attention_mask.to(device)    # video feature에 대한 attention_mask
            target_captions = batch.caption_features.to(device)

            # [BOS] * batch_size
            batch_size = padded_video_feats.size(0)
            decoder_input_ids = torch.ones((batch_size, 1), dtype=torch.long, device=device) * tokenizer.bos_token_id
            #decoder_attention_mask = torch.zeros((batch_size, 1), dtype=torch.long, device=device)
            decoder_attention_mask = torch.ones((batch_size, 1), dtype=torch.long, device=device)
            seed_words = {}
            seed_words["input_ids"] = decoder_input_ids
            seed_words["attention_mask"] = decoder_attention_mask
            #out, past = model(padded_video_feats, attention_mask, target_captions)
            with torch.no_grad():
                # out, _ = model(padded_video_feats, attention_mask, seed_words,
                #                generate_option=True, generate_type="beam")
                generated_token_ids = model(padded_video_feats, attention_mask, seed_words,
                                            generate_option=True, generate_type="beam", **gen_kwargs)

                caps_gen = tokenizer.batch_decode(generated_token_ids, skip_special_tokens=True)
                #print("PAUSE")

            current_result = []
            for i, (gts_i, gen_i, video_id) in enumerate(zip(batch.all_caption_list, caps_gen, batch.video_id_list)):
                #tmp_gen_i = gen_i.split(" ")
                tmp_gen_i = gen_i.strip().split(".")[0]
                gen_i = ' '.join([k for k, g in itertools.groupby([tmp_gen_i])])
                gen['%d_%d' % (it, i)] = [gen_i, ]
                gts['%d_%d' % (it, i)] = gts_i

                tmp_dict = {}
                tmp_dict["video_id"] = video_id
                tmp_dict["generated_caption"] = gen_i
                tmp_dict["ground_truth"] = gts_i

                current_result.append(tmp_dict)

            test_result.extend(current_result)

            pbar.update()

            #break   # debug

            # # with torch.no_grad():
            # #     out, _ = model.beam_search(images, max_len=20, eos_idx=tokenizer.eos_token_id, beam_size=5, out_size=1)
            # caps_gen = tokenizer.decode(generated_token_ids, join_words=False)
            # for i, (gts_i, gen_i) in enumerate(zip(caps_gt, caps_gen)):
            #     gen_i = ' '.join([k for k, g in itertools.groupby(gen_i)])
            #     gen['%d_%d' % (it, i)] = [gen_i, ]
            #     gts['%d_%d' % (it, i)] = gts_i
            # pbar.update()

    gts = evaluation.PTBTokenizer.tokenize(gts)
    gen = evaluation.PTBTokenizer.tokenize(gen)

    scores, _ = evaluation.compute_scores(gts, gen)
    return scores, test_result


def main(args):
    print("  >> Training args: ")
    from pprint import pprint
    pprint(vars(args))

    setproctitle(args.proctitle_name)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    writer = SummaryWriter(log_dir=osp.join(args.logs_folder, args.exp_name))
    logging.basicConfig(filename=args.log_file, level=logging.INFO)
    logging.info(args)

    output_directory = osp.join("saved_models", args.exp_name)
    if not osp.exists(output_directory):
        os.makedirs(output_directory, exist_ok=True)
    print("  >> All the results are stored into {}".format(output_directory))

    config_dict = vars(args)
    with open(osp.join(output_directory, "training_args.json"), "w") as f:
        json.dump(config_dict, f, indent="\t")
    print("  >> Training arguments are stored into {}".format(osp.join(output_directory, "training_args.json")))

    # post-parsing args
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.template = eval(args.template) if type(args.template) is not tuple else args.template

    dataset_path = osp.join(args.data_dir_path, args.dataset_name)
    trainDataset = myDataset(data_dir_path=dataset_path, samples=args.num_target_captions,
                             max_frames=args.max_frames, feature_type=args.feature_type,
                             mode="train", debug_flag=args.debug_flag,
                             long_caption_flag=args.long_caption_flag, num_caption_concat=args.num_caption_concat)

    validDataset = myDataset(data_dir_path=dataset_path, samples=args.num_target_captions,
                             max_frames=args.max_frames, feature_type=args.feature_type,
                             mode="valid", debug_flag=args.debug_flag,
                             long_caption_flag=args.long_caption_flag, num_caption_concat=args.num_caption_concat)

    testDataset = myDataset(data_dir_path=dataset_path, samples=args.num_target_captions,
                            max_frames=args.max_frames, feature_type=args.feature_type,
                            mode="test", debug_flag=args.debug_flag,
                            long_caption_flag=args.long_caption_flag, num_caption_concat=args.num_caption_concat)

    #tokenizer = AutoTokenizer.from_pretrained(args.model_name, add_bos_token=True)

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        outputs = [self.bos_token_id] + token_ids_0 + [self.eos_token_id]
        return outputs
    GPT2Tokenizer.build_inputs_with_special_tokens = build_inputs_with_special_tokens
    tokenizer = GPT2Tokenizer.from_pretrained(args.model_name)
    #tokenizer.build_inputs_with_special_tokens = build_inputs_with_special_tokens
    tokenizer.pad_token = tokenizer.eos_token  # gpt2
    tokenizer.pad_token_id = tokenizer.eos_token_id

    print("  >> Save a tokenizer on {}".format(output_directory))
    tokenizer.save_pretrained(output_directory)


    if args.model_name == "gpt2":
        decoder_hidden_size = 768
    elif args.model_name == "gpt2-medium":
        decoder_hidden_size = 1024
    else:
        raise AssertionError("Invalid model_name: {}".format(args.model_name))

    video_encoder = VideoEncoder(args=args, decoder_hidden_size= decoder_hidden_size,
                                 max_frames=args.max_frames, device=device)

    model = MyGPT2Decoder(args=args, encoder = video_encoder, bos_idx=tokenizer.bos_token_id, tokenizer=tokenizer)
    # if torch.cuda.device_count() > 1:
    #     print("Let's use", torch.cuda.device_count(), "GPUs!")
    #     model = torch.nn.DataParallel(model)

    model = model.to(device)
    #model = myVideoCaptioningModel()
    gpt_optimizer = AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8)

    loss_fn = NLLLoss(ignore_index=-100)

    dataloader_train = DataLoader(trainDataset, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, drop_last=True, pin_memory=True,
                                  collate_fn=myDataCollatorForSeq2Seq(tokenizer=tokenizer,
                                                                      max_length=args.max_sequence_length))

    dataloader_valid = DataLoader(validDataset, batch_size=args.batch_size, shuffle=False,
                                  num_workers=args.num_workers, drop_last=False, pin_memory=True,
                                  collate_fn=myDataCollatorForSeq2Seq(tokenizer=tokenizer,
                                                                      max_length=args.max_sequence_length))

    dataloader_test = DataLoader(testDataset, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.num_workers, drop_last=False, pin_memory=True,
                                 collate_fn=myDataCollatorForSeq2Seq(tokenizer=tokenizer,
                                                                     max_length=args.max_sequence_length))

    gen_kwargs = {
        "max_length" : args.gen_sequence_length,
        "num_beams" : args.num_beams,
        "early_stopping": args.gen_early_stopping
    }

    start_epoch = 0
    use_rl = False
    best_cider = 0.0
    patience = 0
    best_epoch = -1

    for e in range(start_epoch, start_epoch + 100):

        # per-epoch train loss
        train_loss = train_xe(model, dataloader=dataloader_train, gpt_optimizer=gpt_optimizer,
                              dataloader_eval=dataloader_valid, args=args, device=device,
                              loss_fn=loss_fn, tokenizer=tokenizer, cur_epoch=e)

        writer.add_scalar("data/train_loss", train_loss, global_step=e)

        # validation loss for every epoch
        val_loss = evaluate_loss(model=model, dataloader=dataloader_valid, loss_fn=loss_fn,
                                 cur_epoch=e, device=device, tokenizer=tokenizer)
        writer.add_scalar("data/val_loss", val_loss, global_step=e)


        # validation scores
        scores, _  = evaluate_metrics(model=model, dataloader=dataloader_valid, tokenizer=tokenizer,
                                      exp_name=args.exp_name+"_val", epoch=e, device=device,
                                      **gen_kwargs)

        val_cider = scores['CIDEr']
        writer.add_scalar('data/val_cider', val_cider, e)
        writer.add_scalar('data/val_bleu1', scores['BLEU'][0], e)
        writer.add_scalar('data/val_bleu4', scores['BLEU'][3], e)
        writer.add_scalar('data/val_meteor', scores['METEOR'], e)
        writer.add_scalar('data/val_rouge', scores['ROUGE'], e)

        logging.info("val cider\t" + str(val_cider) + "current epoch " + str(e))
        logging.info("val bleu1\t" + str(scores["BLEU"][0]) + "current epoch " + str(e))
        logging.info("val bleu4\t" + str(scores["BLEU"][3]) + "current epoch " + str(e))
        logging.info("val meteor\t" + str(scores["METEOR"]) + "current epoch " + str(e))
        logging.info("val rouge\t" + str(scores["ROUGE"]) + "current epoch " + str(e))


        # Test scores
        test_scores, test_result  = evaluate_metrics(model, dataloader=dataloader_test, tokenizer=tokenizer,
                                                     exp_name=args.exp_name + "_test", epoch=e, device=device,
                                                     **gen_kwargs)

        writer.add_scalar('data/test_cider', test_scores['CIDEr'], e)
        writer.add_scalar('data/test_bleu1', test_scores['BLEU'][0], e)
        writer.add_scalar('data/test_bleu4', test_scores['BLEU'][3], e)
        writer.add_scalar('data/test_meteor', test_scores['METEOR'], e)
        writer.add_scalar('data/test_rouge', test_scores['ROUGE'], e)

        logging.info("test cider\t" + str(test_scores['CIDEr']) + "current epoch " + str(e))
        logging.info("test bleu1\t" + str(test_scores["BLEU"][0]) + "current epoch " + str(e))
        logging.info("test bleu4\t" + str(test_scores["BLEU"][3]) + "current epoch " + str(e))
        logging.info("test meteor\t" + str(test_scores["METEOR"]) + "current epoch " + str(e))
        logging.info("test rouge\t" + str(test_scores["ROUGE"]) + "current epoch " + str(e))

        best = False
        if val_cider >= best_cider:
            best_cider = val_cider
            patience = 0
            best = True
            best_epoch = e
        else:
            patience += 1

        switch_to_rl = False
        exit_train = False

        if patience == 7:
            from pprint import pprint
            print("\n  >> Current validation scores on epoch - {}".format(e))
            pprint(scores)
            print("\n  >> Current test scores on epoch - {}".format(e))
            pprint(test_scores)
            print("\n  >> Best record: epoch-{} / best_cider - {}".format(e, best_cider))
            print("  >> Current patience: {}".format(patience))
            print("\n")
            print("  >> Early-stopping.... ")

            logging.info("  >> Best record: epoch-{} / best_cider - {}".format(best_epoch, best_cider))
            logging.info("  >> Current patience: {}".format(patience))

            break

        # if patience == 5:
        #     if not use_rl:
        #         use_rl = True
        #         switch_to_rl = True
        #         patience = 0
        #
        #
        #         gpt_optimizer = AdamW(model.parameters(),
        #                              lr = args.reinforcement_lr,betas=(0.9, 0.999), eps=1e-8)
        #
        #         print("Switching to RL")
        #     else:
        #         print('patience reached.')
        #         exit_train = True
        #
        # if switch_to_rl and not best:
        #     print(" now we are resuming!!!!")
        #     data = torch.load('saved_models/%s_best.pth' % args.exp_name)
        #     torch.set_rng_state(data['torch_rng_state'])
        #     torch.cuda.set_rng_state(data['cuda_rng_state'])
        #     np.random.set_state(data['numpy_rng_state'])
        #     random.setstate(data['random_rng_state'])
        #     model.load_state_dict(data['state_dict'])
        #     print('Resuming from epoch %d, validation loss %f, and best cider %f' % (
        #         data['epoch'], data['val_loss'], data['best_cider']))

        output_dir_path = osp.join("saved_models", args.exp_name, "epoch_"+str(e))
        if not osp.exists(output_dir_path):
            os.makedirs(output_dir_path, exist_ok=True)
        save_pth_name = osp.join(output_dir_path, "save.pth")
        torch.save({
            'torch_rng_state': torch.get_rng_state(),
            'cuda_rng_state': torch.cuda.get_rng_state(),
            'numpy_rng_state': np.random.get_state(),
            'random_rng_state': random.getstate(),
            'epoch': e,
            'val_loss': val_loss,
            'val_cider': val_cider,
            'state_dict': model.state_dict(),
            'optimizer': gpt_optimizer.state_dict(),
            'patience': patience,
            'best_cider': best_cider,
            'use_rl': use_rl,
        #}, 'saved_models/%s_last.pth' % args.exp_name)
        }, save_pth_name)

        with open(osp.join(output_dir_path, "test_result.json"), "w") as f:
            json.dump(scores, f, indent="\t")
            json.dump(test_scores, f, indent = "\t")
            json.dump(test_result, f, indent = "\t")

        if best:
            best_dir_path = osp.join("saved_models", args.exp_name, "best")
            if not osp.exists(best_dir_path):
                os.makedirs(best_dir_path, exist_ok=True)

            copyfile(save_pth_name, osp.join(best_dir_path, "best.pth"))
            copyfile(osp.join(output_dir_path, "test_result.json"),
                     osp.join(best_dir_path, "best_test_result.json"))

        from pprint import pprint
        print("\n  >> Current validation scores on epoch - {}".format(e))
        pprint(scores)
        print("\n  >> Current test scores on epoch - {}".format(e))
        pprint(test_scores)
        print("  >> Sample video data: {}".format(test_result[0]["video_id"]))
        print("     > Generated caption: {}".format(test_result[0]["generated_caption"]))
        print("     > Ground Truth: ")
        pprint(test_result[0]["ground_truth"][0:5])
        print("\n")

        print("\n  >> Best record: epoch-{} / best_cider - {}".format(e, best_cider))
        print("  >> Current patience: {}".format(patience))
        print("\n")

        logging.info("  >> Best record: epoch-{} / best_cider - {}".format(e, best_cider))
        logging.info("  >> Current patience: {}".format(patience))

    print("  >> End of training....")

    return

if __name__ == "__main__":
    args = get_parser().parse_args()

    main(args)
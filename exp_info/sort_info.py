
import argparse
import pandas as pd

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--exp_info_file", type=str, default="exp_info", help="Where to save the model once it is trained.")
args = parser.parse_args()

df = pd.read_csv(f"./{args.exp_info_file}", sep = "\t", engine='python', encoding = "cp949", header=None)
df.rename(columns={0: "lambda", 1: "dir"}, inplace=True)
df = df.sort_values(by="lambda", ascending=False)
df.to_csv(f"./{args.exp_info_file}", sep = '\t', index = False, header=False)


# f = open(f"./exp_info/{args.exp_info_file}.txt", 'a')
# f.write(f"{3}\t afsfsda\n")
# f.close()
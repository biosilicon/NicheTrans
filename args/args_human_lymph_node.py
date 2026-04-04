import argparse
from utils.moe_config import parse_literal_value

def generate_args():
    parser = argparse.ArgumentParser(description='Multi-omics translation')
    ################
    # for model
    parser.add_argument('--noise_rate', default=0.5, type=float)
    parser.add_argument('--dropout_rate', default=0.1, type=float)
    parser.add_argument('--use_moe_ffn', default=True,
                        type=lambda x: str(x).lower() in ('1', 'true', 'yes', 'y'))
    parser.add_argument('--num_experts', default=4, type=parse_literal_value)
    parser.add_argument('--moe_gate_hidden_dim', default=256, type=parse_literal_value)
    parser.add_argument('--moe_gate_type', default='softmax', type=parse_literal_value)
    parser.add_argument('--ffn_mult', default=4, type=parse_literal_value)
    parser.add_argument('--moe_num_layers', default=1, type=int)
    parser.add_argument('--moe_router_temperature_enable', default=True, type=parse_literal_value)
    parser.add_argument('--moe_router_temperature_start', default=1.0, type=parse_literal_value)
    parser.add_argument('--moe_router_temperature_mid', default=0.7, type=parse_literal_value)
    parser.add_argument('--moe_router_temperature_end', default=0.5, type=parse_literal_value)
    parser.add_argument('--moe_router_temperature_schedule', default='step', type=parse_literal_value)
    parser.add_argument('--moe_balance_loss_enable', default=True, type=parse_literal_value)
    parser.add_argument('--moe_balance_loss_weight', default=3e-3, type=parse_literal_value)
    parser.add_argument('--moe_balance_loss_type', default='mse_uniform', type=parse_literal_value)
    parser.add_argument('--moe_router_entropy_penalty_enable', default=True, type=parse_literal_value)
    parser.add_argument('--moe_router_entropy_penalty_weight', default=0.003, type=parse_literal_value)

    # Datasets
    parser.add_argument('--n_source', default=3000, type=int)
    parser.add_argument('-j', '--workers', default=4, type=int,
                        help="number of data loading workers (default: 4)")
    
    parser.add_argument('--adata_path', default='/mnt/datadisk0/Processed_DATA/2024_nm_human_lymph_nodes/', type=str)

    # Training options
    parser.add_argument('--max-epoch', default=10, type=int,
                        help="maximum epochs to run")
    parser.add_argument('--stepsize', default=10, type=int,
                        help="stepsize to decay learning rate (>0 means this is enabled)")

    parser.add_argument('--train-batch', default=32, type=int)
    parser.add_argument('--test-batch', default=32, type=int)

    # Optimization options
    parser.add_argument('--optimizer', default='adam', type=str,
                        help="adam or SGD")
    parser.add_argument('--lr', '--learning-rate', default=0.0003, type=float,
                        help="initial learning rate, use 0.0001")
    parser.add_argument('--gamma', default=0.1, type=float,
                        help="learning rate decay")
    parser.add_argument('--weight-decay', default=5e-04, type=float,
                        help="weight decay (default: 5e-04)")

    # Miscs
    parser.add_argument('--seed', type=int, default=1, help="manual seed")
    parser.add_argument('--save-dir', type=str, default='./log', help="manual seed")
    parser.add_argument('--eval-step', type=int, default=1,
                        help="run evaluation for every N epochs (set to -1 to test after training)")
    parser.add_argument('--gpu-devices', default='0', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = generate_args()

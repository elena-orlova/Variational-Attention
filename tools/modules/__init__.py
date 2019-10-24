from tools.modules.UtilClass import LayerNorm, Elementwise
from tools.modules.Gate import context_gate_factory, ContextGate
from tools.modules.GlobalAttention import GlobalAttention
from tools.modules.VariationalAttention import VariationalAttention, Params
from tools.modules.ConvMultiStepAttention import ConvMultiStepAttention
from tools.modules.CopyGenerator import CopyGenerator, CopyGeneratorLossCompute
from tools.modules.StructuredAttention import MatrixTree
from tools.modules.Transformer import \
   TransformerEncoder, TransformerDecoder, PositionwiseFeedForward
from tools.modules.Conv2Conv import CNNEncoder, CNNDecoder
from tools.modules.MultiHeadedAttn import MultiHeadedAttention
from tools.modules.StackedRNN import StackedLSTM, StackedGRU
from tools.modules.Embeddings import Embeddings, PositionalEncoding
from tools.modules.WeightNorm import WeightNormConv2d

from tools.Models import EncoderBase, MeanEncoder, StdRNNDecoder, \
    RNNDecoderBase, InputFeedRNNDecoder, RNNEncoder, NMTModel

from tools.modules.SRU import check_sru_requirement
can_use_sru = check_sru_requirement()
if can_use_sru:
    from tools.modules.SRU import SRU


# For flake8 compatibility.
__all__ = [EncoderBase, MeanEncoder, RNNDecoderBase, InputFeedRNNDecoder,
           RNNEncoder, NMTModel,
           StdRNNDecoder, ContextGate, GlobalAttention,
           VariationalAttention,
           PositionwiseFeedForward, PositionalEncoding,
           CopyGenerator, MultiHeadedAttention,
           LayerNorm,
           TransformerEncoder, TransformerDecoder, Embeddings, Elementwise,
           MatrixTree, WeightNormConv2d, ConvMultiStepAttention,
           CNNEncoder, CNNDecoder, StackedLSTM, StackedGRU,
           context_gate_factory, CopyGeneratorLossCompute]

if can_use_sru:
    __all__.extend([SRU, check_sru_requirement])

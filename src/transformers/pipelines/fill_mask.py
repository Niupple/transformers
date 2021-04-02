from typing import TYPE_CHECKING, Optional, Union

import numpy as np

from ..file_utils import add_end_docstrings, is_tf_available, is_torch_available
from ..modelcard import ModelCard
from ..tokenization_utils import PreTrainedTokenizer
from ..utils import logging
from .base import PIPELINE_INIT_ARGS, ArgumentHandler, Pipeline, PipelineException


if TYPE_CHECKING:
    from ..modeling_tf_utils import TFPreTrainedModel
    from ..modeling_utils import PreTrainedModel

if is_tf_available():
    import tensorflow as tf

    from ..models.auto.modeling_tf_auto import TF_MODEL_WITH_LM_HEAD_MAPPING

if is_torch_available():
    import torch
    import torch.multiprocessing as mp
    from ..models.auto.modeling_auto import MODEL_FOR_MASKED_LM_MAPPING


logger = logging.get_logger(__name__)
sentinal = None

@add_end_docstrings(
    PIPELINE_INIT_ARGS,
    r"""
        top_k (:obj:`int`, defaults to 5): The number of predictions to return.
    """,
)
class FillMaskPipeline(Pipeline):
    """
    Masked language modeling prediction pipeline using any :obj:`ModelWithLMHead`. See the `masked language modeling
    examples <../task_summary.html#masked-language-modeling>`__ for more information.

    This mask filling pipeline can currently be loaded from :func:`~transformers.pipeline` using the following task
    identifier: :obj:`"fill-mask"`.

    The models that this pipeline can use are models that have been trained with a masked language modeling objective,
    which includes the bi-directional models in the library. See the up-to-date list of available models on
    `huggingface.co/models <https://huggingface.co/models?filter=masked-lm>`__.

    .. note::

        This pipeline only works for inputs with exactly one token masked.
    """

    def __init__(
        self,
        model: Union["PreTrainedModel", "TFPreTrainedModel"],
        tokenizer: PreTrainedTokenizer,
        modelcard: Optional[ModelCard] = None,
        framework: Optional[str] = None,
        args_parser: ArgumentHandler = None,
        device: int = -1,
        top_k=5,
        task: str = "",
        data_parallel: bool = False
    ):
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            modelcard=modelcard,
            framework=framework,
            args_parser=args_parser,
            device=device,
            binary_output=True,
            task=task,
            data_parallel=data_parallel
        )

        self.check_model_type(TF_MODEL_WITH_LM_HEAD_MAPPING if self.framework == "tf" else MODEL_FOR_MASKED_LM_MAPPING)
        self.top_k = top_k
        self.scripted = False

    def ensure_exactly_one_mask_token(self, masked_index: np.ndarray):
        numel = np.prod(masked_index.shape)
        if numel > 1:
            raise PipelineException(
                "fill-mask",
                self.model.base_model_prefix,
                f"More than one mask_token ({self.tokenizer.mask_token}) is not supported",
            )
        elif numel < 1:
            raise PipelineException(
                "fill-mask",
                self.model.base_model_prefix,
                f"No mask_token ({self.tokenizer.mask_token}) found on the input",
            )

    def _stage_1(self, *args, **kwargs):
        return self._parse_and_tokenize(*args, **kwargs)

    def _stage_2(self, inputs, *args, **kwargs):
        return self._forward(inputs, return_tensors=True, move_to_cpu=False)

    def _stage_3(self, inputs, outputs, *args, targets=None, top_k: Optional[int] = None, **kwargs):
        results = []
        batch_size = outputs.shape[0] if self.framework == "tf" else outputs.size(0)

        if targets is not None:
            if len(targets) == 0 or len(targets[0]) == 0:
                raise ValueError("At least one target must be provided when passed.")
            if isinstance(targets, str):
                targets = [targets]

            targets_proc = []
            for target in targets:
                target_enc = self.tokenizer.tokenize(target)
                if len(target_enc) > 1 or target_enc[0] == self.tokenizer.unk_token:
                    logger.warning(
                        "The specified target token `{}` does not exist in the model vocabulary. Replacing with `{}`.".format(
                            target, target_enc[0]
                        )
                    )
                targets_proc.append(target_enc[0])
            target_inds = np.array(self.tokenizer.convert_tokens_to_ids(targets_proc))

        # much better
        if self.framework == 'pt':
            logits = torch.empty((batch_size, outputs.shape[2]), dtype=outputs.dtype, device=torch.device('cuda'))
            input_ids = inputs["input_ids"]
            masked_index = torch.nonzero(input_ids.eq(self.tokenizer.mask_token_id))[..., 1].tolist()
            # print(masked_index)
            # the best to remove loops altogether, maybe requies a custom op
            for i in range(batch_size):
                # print(logits.shape, masked_index[i])
                logits[i, :] = outputs[i, masked_index[i], :]
            probs = logits.softmax(dim=1)
            if targets is None:
                values, predictions = probs.topk(top_k if top_k is not None else self.top_k, dim=1)
                values = values.cpu()
                predictions = predictions.cpu()
            else:
                values = probs[..., target_inds].cpu()
                sort_inds = list(reversed(values.argsort(dim=-1)))
                values = values[..., sort_inds]
                predictions = target_inds[sort_inds]

        # tensorflow legacy
        if self.framework == "tf":
            for i in range(batch_size):
                input_ids = inputs["input_ids"][i]
                result = []
                masked_index = tf.where(input_ids == self.tokenizer.mask_token_id).numpy()

                # Fill mask pipeline supports only one ${mask_token} per sample
                self.ensure_exactly_one_mask_token(masked_index)

                logits = outputs[i, masked_index.item(), :]
                probs = tf.nn.softmax(logits)
                if targets is None:
                    topk = tf.math.top_k(probs, k=top_k if top_k is not None else self.top_k)
                    values, predictions = topk.values.numpy(), topk.indices.numpy()
                else:
                    values = tf.gather_nd(probs, tf.reshape(target_inds, (-1, 1)))
                    sort_inds = tf.reverse(tf.argsort(values), [0])
                    values = tf.gather_nd(values, tf.reshape(sort_inds, (-1, 1))).numpy()
                    predictions = target_inds[sort_inds.numpy()]

        for i in range(batch_size):
            input_ids = inputs["input_ids"][i]
            result = []
            for v, p in zip(values[i].tolist(), predictions[i].tolist()):
                tokens = input_ids.numpy()
                tokens[masked_index] = p
                # Filter padding out:
                tokens = tokens[np.where(tokens != self.tokenizer.pad_token_id)]
                result.append(
                    {
                        "sequence": self.tokenizer.decode(tokens, skip_special_tokens=True),
                        "score": v,
                        "token": p,
                        "token_str": self.tokenizer.decode(p),
                    }
                )

            # Append
            results += [result]

        torch.cuda.synchronize()

        if len(results) == 1:
            return results[0]
        return results

    def __call__(self, *args, targets=None, top_k: Optional[int] = None, **kwargs):
        """
        Fill the masked token in the text(s) given as inputs.

        Args:
            args (:obj:`str` or :obj:`List[str]`):
                One or several texts (or one list of prompts) with masked tokens.
            targets (:obj:`str` or :obj:`List[str]`, `optional`):
                When passed, the model will return the scores for the passed token or tokens rather than the top k
                predictions in the entire vocabulary. If the provided targets are not in the model vocab, they will be
                tokenized and the first resulting token will be used (with a warning).
            top_k (:obj:`int`, `optional`):
                When passed, overrides the number of predictions to return.

        Return:
            A list or a list of list of :obj:`dict`: Each result comes as list of dictionaries with the following keys:

            - **sequence** (:obj:`str`) -- The corresponding input with the mask token prediction.
            - **score** (:obj:`float`) -- The corresponding probability.
            - **token** (:obj:`int`) -- The predicted token id (to replace the masked one).
            - **token** (:obj:`str`) -- The predicted token (to replace the masked one).
        """
        import time
        inputs = self._stage_1(*args, **kwargs)
        # print("inputs = ", inputs)
        outputs = self._stage_2(inputs)
        results = self._stage_3(inputs, outputs, targets, top_k)

        return results

    def w_stage_1(self, qin, qout):
        while True:
            x = qin.get()
            if x is sentinal:
                # print("s1 met sentinal, waiting")
                qout.put(sentinal)
                self.p2_exited.wait()
                break
            args, kwargs = x
            # print("s1 got a task.")
            y = self._stage_1(*args, **kwargs)
            # print("s1 put y")
            qout.put(y)
        # print("s1 exiting")

    def w_stage_2(self, qin, qout):
        while True:
            inputs = qin.get()
            if inputs is sentinal:
                # print("s2 met sentinal, waiting")
                qout.put(sentinal)
                self.p3_exited.wait()
                break
            # print("s2 got a task")
            outputs = self._stage_2(inputs)
            inputs_clone = inputs
            # print("s2 put")
            qout.put((inputs_clone, outputs))
        self.p2_exited.set()
        # print("s2 exiting")

    def w_stage_3(self, qin, qout):
        while True:
            x = qin.get()
            if x is sentinal:
                # print("s3 met sentinal, exiting")
                break
            inputs, outputs = x
            # print("s3 got a job")
            result = self._stage_3(inputs, outputs) # parameters like targets and topk cannot be customized
            del inputs, outputs
            # print("s3 put")
            qout.put(result)
        self.p3_exited.set()
        qout.put(sentinal)

    def infer_offline(self, works: list, *args, max_in_queue=5, **kwargs):
        mp.set_start_method('forkserver')
        qin = mp.Queue()
        q1 = mp.Queue()
        q2 = mp.Queue()
        qout = mp.Queue()

        self.p3_exited = mp.Event()
        self.p2_exited = mp.Event()

        p1 = mp.Process(target=self.w_stage_1, args=(qin, q1))
        p2 = mp.Process(target=self.w_stage_2, args=(q1, q2))
        p3 = mp.Process(target=self.w_stage_3, args=(q2, qout))
        
        p1.start()
        p2.start()
        p3.start()

        for work in works:
            qin.put(([work], dict()))
        qin.put(sentinal)
        results = []
        while True:
            x = qout.get()
            if x is sentinal:
                break
            else:
                results.append(x)
        p1.join()
        p2.join()
        p3.join()

        return results

    # not great at all
    def infer_offline_thread(self, works):
        import threading as mt
        from queue import Queue

        qin = Queue()
        q1 = Queue()
        q2 = Queue()
        qout = Queue()

        self.p3_exited = mt.Event()
        self.p2_exited = mt.Event()

        p1 = mt.Thread(target=self.w_stage_1, args=(qin, q1))
        p2 = mt.Thread(target=self.w_stage_2, args=(q1, q2))
        p3 = mt.Thread(target=self.w_stage_3, args=(q2, qout))
        
        p1.start()
        p2.start()
        p3.start()

        for work in works:
            qin.put(([work], dict()))
        qin.put(sentinal)
        results = []
        while True:
            x = qout.get()
            if x is sentinal:
                break
            else:
                results.append(x)
        p1.join()
        p2.join()
        p3.join()

        return results

from typing import List, Optional, Any
import data_structs as ds
import logging
import general_utils

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s')


def paraphrase_questions(
    question: str, 
    k: int, 
    llm: Any, 
    utils: Any,
    max_return_per_call: int = 32
) -> List[str]:
    
    logging.info("Starting paraphrase_questions for question: %s", question)
    rephrase_questions = []
    full_batches = k // max_return_per_call
    remainder = k % max_return_per_call
    prompt = utils.paraphrase_prompt.format(q=question)
    for batch in range(full_batches):
        batch_output = llm.generate(prompt=prompt, num_return_sequences=max_return_per_call).text
        rephrase_questions += batch_output
        logging.info("Paraphrase batch %d/%d generated %d questions", batch+1, full_batches, len(batch_output))
    # Remainder batch
    if remainder:
        batch_output = llm.generate(prompt=prompt, num_return_sequences=remainder).text
        rephrase_questions += batch_output
        logging.info("Paraphrase remainder batch generated %d questions", len(batch_output))
    logging.info("Generated %d rephrased questions:\n %s", len(rephrase_questions), rephrase_questions)
    return rephrase_questions

def paraphrase_batch(
    questions: List[str], 
    llm: Any,
    utils: Any
) -> List[ds.QuestionNode]:

    logging.info("Starting paraphrase batch for %d questions", len(questions))
    prompts = [utils.paraphrase_prompt.format(q=q) for q in questions]
    paraphrased_gens = llm.batch_generate(prompts=prompts, num_return_sequences=1)
    paraphrased_qs = [ds.QuestionNode(question=gen.text[0]) for gen in paraphrased_gens]
    logging.info("Paraphrased %d questions", len(paraphrased_qs))
    return paraphrased_qs


def clarify_batch(
    questions: List[str], 
    llm: Any,
    utils: Any
) -> List[ds.QuestionNode]:

    logging.info("Starting clarify_batch for %d questions", len(questions))
    prompts = [utils.clarify_prompt.format(q=q) for q in questions]
    clarify_gens = llm.batch_generate(prompts=prompts, num_return_sequences=1)
    clarify_qs = [ds.QuestionNode(question=gen.text[0]) for gen in clarify_gens]
    logging.info("Clarified %d questions", len(clarify_qs))
    return clarify_qs


def first_trial_batch(
    questions: List[str], 
    gt: str,
    llm: Any, 
    utils: Any,
    choice_text: Optional[str], 
    passage: Optional[str]
) -> List[ds.AnswerNode]:

    logging.info("Starting first_trial_batch")
    if utils.name in general_utils.MCQ_UTILS:
        prompts = [utils.sampling_prompt.format(q=q, c=choice_text) for q in questions]
    elif utils.name in general_utils.RCQ_UTILS:
        prompts = [utils.sampling_prompt.format(q=q, p=passage) for q in questions]
    else:
        prompts = [utils.sampling_prompt.format(q=q) for q in questions]

    sampling_gens = llm.batch_generate(prompts=prompts, num_return_sequences=1)
    sampling_answers = [
        ds.AnswerNode(
            raw_answer=gen.text[0],
            extract_answer=utils.extract_answer(gen.text[0]),
            is_correct=utils.eval_answer(gen.text[0], gt, question),
            token_probs=gen.log_prob[0]
        )
        for gen, question in zip(sampling_gens, questions)
    ]
    return sampling_answers


def iterative_trial_batch(
    questions: List[str], 
    ans_seqs: List[List[str]],
    gt: str,
    llm: Any, 
    utils: Any,
    choice_text: Optional[str], 
    passage: Optional[str]
) -> List[ds.AnswerNode]:

    logging.info("Starting iterative_trial")
    ans_seq = [
        "\n".join([f"Solution{i+1}: {a}" for i, a in enumerate(seq)])
        for seq in ans_seqs
    ]
    if utils.name in general_utils.MCQ_UTILS:
        prompts = [
            utils.iterative_prompt.format(q=q, c=choice_text, aseq=aseq) 
            for q, aseq in zip(questions, ans_seq)]
    elif utils.name in general_utils.RCQ_UTILS:
        prompts = [
            utils.iterative_prompt.format(q=q, p=passage, aseq=aseq) 
            for q, aseq in zip(questions, ans_seq)]
    else:
        prompts = [
            utils.iterative_prompt.format(q=q, aseq=aseq) 
            for q, aseq in zip(questions, ans_seq)]
    # ipdb.set_trace()
    sampling_gens = llm.batch_generate(prompts=prompts, num_return_sequences=1)
    sampling_answers = [
        ds.AnswerNode(
            raw_answer=gen.text[0],
            extract_answer=utils.extract_answer(gen.text[0]),
            is_correct=utils.eval_answer(gen.text[0], gt, question),
            token_probs=gen.log_prob[0]
        )
        for gen, question in zip(sampling_gens, questions)
    ]
    return sampling_answers


def self_check_batch(
    questions: List[str], 
    first_trial_answers: List[str],
    gt: str,
    llm: Any, 
    utils: Any,
    choice_text: Optional[str], 
    passage: Optional[str] 
) -> List[ds.AnswerNode]:

    logging.info("Starting self_check_batch for %d questions", len(questions))

    if utils.name in general_utils.MCQ_UTILS:
        prompts = [
            utils.check_prompt.format(q=q, a=a, c=choice_text)
            for q, a in zip(questions, first_trial_answers)
        ]
    elif utils.name in general_utils.RCQ_UTILS:
        prompts = [
            utils.check_prompt.format(q=q, a=a, p=passage)
            for q, a in zip(questions, first_trial_answers)
        ]
    else:
        prompts = [
            utils.check_prompt.format(q=q, a=a)
            for q, a in zip(questions, first_trial_answers)
        ]
    self_check_gens = llm.batch_generate(prompts=prompts, num_return_sequences=1)
    self_check_answers = [
        ds.AnswerNode(
            raw_answer=gen.text[0],
            extract_answer=utils.extract_answer(gen.text[0]),
            is_correct=utils.eval_answer(gen.text[0], gt, question),
            token_probs=gen.log_prob[0]
        )
        for gen, question in zip(self_check_gens, questions)
    ]
    logging.info("Self-check batch produced %d answers", len(self_check_answers))
    return self_check_answers


def generate_batch_chain(
    questions: List[str], 
    ground_truth: str, 
    llm: Any, 
    utils: Any,
    choice_text: Optional[str],
    passage: Optional[str] 
) -> List[ds.Chain]:

    logging.info("Starting generate chain for a batch of %d questions", len(questions))
    # For clarity, we may want to pass the actual question text from the clarified nodes
    clarify_qs = clarify_batch(questions, llm, utils)
    # Extract the clarified question strings (assuming each QuestionNode has a .question property)
    clarified_texts = [node.question for node in clarify_qs]
    
    first_trial_answers = first_trial_batch(
        questions=clarified_texts, 
        gt=ground_truth, llm=llm, utils=utils,
        choice_text=choice_text, passage=passage
    )
    checked_answers = self_check_batch(
        questions=clarified_texts, 
        first_trial_answers=[ans.raw_answer for ans in first_trial_answers], 
        gt=ground_truth, llm=llm, utils=utils, 
        choice_text=choice_text, passage=passage
    )
    
    batch_chains = []
    for orig_q, clarify_q, first_trial_answer, checked_answer in zip(questions, clarify_qs, first_trial_answers, checked_answers):
        chain = ds.Chain(
            paraphrased_question=ds.QuestionNode(question=orig_q),
            clarified_question=clarify_q,
            first_trial_answer=first_trial_answer,
            self_checked_answer=checked_answer,
        )
        batch_chains.append(chain)
        # Print text outputs for debugging
        print("------ Chain Generated ------")
        print("Original Paraphrased Question:", orig_q)
        print("Clarified Question:", clarify_q.question)
        print("First Trial Answer:", first_trial_answer.raw_answer)
        print("Self-Checked Answer:", checked_answer.raw_answer)
        print("Correctness:", checked_answer.is_correct)
        print("-----------------------------\n")
    logging.info("Generated %d chains for this batch", len(batch_chains))
    return batch_chains


def generate_au_test_chain(
    questions: List[str], 
    ground_truth: str, 
    llm: Any, 
    utils: Any,
    choice_text: Optional[str],
    passage: Optional[str] 
) -> List[ds.Chain]:

    logging.info("Starting generate chain for a batch of %d questions", len(questions))
    # For clarity, we may want to pass the actual question text from the clarified nodes
    clarify_qs = clarify_batch(questions, llm, utils)
    # Extract the clarified question strings (assuming each QuestionNode has a .question property)
    clarified_texts = [node.question for node in clarify_qs]
    
    first_trial_answers = first_trial_batch(
        questions=clarified_texts, 
        gt=ground_truth, llm=llm, utils=utils,
        choice_text=choice_text, passage=passage)
    
    batch_chains = []
    for clarify_q, first_trial_answer in zip(clarify_qs, first_trial_answers):
        chain = ds.Chain(
            paraphrased_question=clarify_q,
            clarified_question=clarify_q,
            first_trial_answer=first_trial_answer,
            self_checked_answer=first_trial_answer,
        )
        batch_chains.append(chain)
        # Print text outputs for debugging
        print("------ Chain Generated ------")
        print("Clarified Question:", clarify_q.question)
        print("Chain Answer:", first_trial_answer.raw_answer)
        print("Chain Correctness:", first_trial_answer.is_correct)
        print("-----------------------------\n")
    logging.info("Generated %d chains for this batch", len(batch_chains))
    return batch_chains


def generate_su_test_chain(
    questions: List[str], 
    ground_truth: str, 
    llm: Any, 
    utils: Any,
    choice_text: Optional[str],
    passage: Optional[str] 
) -> List[ds.Chain]:

    logging.info("Starting generate chain for a batch of %d questions", len(questions))
    # For clarity, we may want to pass the actual question text from the clarified nodes
    paraphrase_qs = paraphrase_batch(questions, llm, utils)
    # Extract the paraphrased question strings (assuming each QuestionNode has a .question property)
    paraphrase_texts = [node.question for node in paraphrase_qs]
    
    first_trial_answers = first_trial_batch(
        questions=paraphrase_texts, 
        gt=ground_truth, llm=llm, utils=utils,
        choice_text=choice_text, passage=passage)
    
    batch_chains = []
    for paraphrase_q, first_trial_answer in zip(paraphrase_qs, first_trial_answers):
        chain = ds.Chain(
            paraphrased_question=paraphrase_q,
            clarified_question=paraphrase_q,
            first_trial_answer=first_trial_answer,
            self_checked_answer=first_trial_answer,
        )
        batch_chains.append(chain)
        # Print text outputs for debugging
        print("------ Chain Generated ------")
        print("Paraphrased Question:", paraphrase_q.question)
        print("Chain Answer:", first_trial_answer.raw_answer)
        print("Chain Correctness:", first_trial_answer.is_correct)
        print("-----------------------------\n")
    logging.info("Generated %d chains for this batch", len(batch_chains))
    return batch_chains


def generate_batch_eu_ou_test_chain(
    questions: List[str], 
    ground_truth: str, 
    llm: Any, 
    utils: Any,
    choice_text: Optional[str],
    passage: Optional[str] 
) -> List[ds.Iterative_Chain]:

    logging.info("Starting generate iterative_chain for a batch of %d questions", len(questions))
    first_trial_answer_nodes = first_trial_batch(
        questions=questions, gt=ground_truth, 
        llm=llm, utils=utils, choice_text=choice_text, passage=passage)
    
    level_nodes = [first_trial_answer_nodes]
    ans_seqs = [node.raw_answer for node in first_trial_answer_nodes]
    checked_answer_nodes = self_check_batch(
        questions=questions, first_trial_answers=ans_seqs, 
        gt=ground_truth, llm=llm, utils=utils, 
        choice_text=choice_text, passage=passage
        )
    level_nodes.append(checked_answer_nodes)
    batch_iterative_chains = [
        ds.Iterative_Chain(answer_node_seq=list(col)) for col in zip(*level_nodes)]
    return batch_iterative_chains


def generate_batch_iterative_chain(
    questions: List[str], 
    ground_truth: str, 
    round: int,
    llm: Any, 
    utils: Any,
    choice_text: Optional[str],
    passage: Optional[str] 
) -> List[ds.Iterative_Chain]:

    logging.info("Starting generate iterative_chain for a batch of %d questions", len(questions))
    first_trial_answer_nodes = first_trial_batch(
        questions=questions, gt=ground_truth, 
        llm=llm, utils=utils, choice_text=choice_text, passage=passage)
    
    level_nodes = [first_trial_answer_nodes]
    ans_seqs = [[node.raw_answer] for node in first_trial_answer_nodes]
    for _ in range(round):
        next_trial_answer_nodes = iterative_trial_batch(
            questions=questions, ans_seqs=ans_seqs, gt=ground_truth, 
            llm=llm, utils=utils, choice_text=choice_text, passage=passage)
        level_nodes.append(next_trial_answer_nodes)
        for idx, node in enumerate(next_trial_answer_nodes):
            ans_seqs[idx].append(node.raw_answer)
    batch_iterative_chains = [ds.Iterative_Chain(answer_node_seq=list(col)) for col in zip(*level_nodes)]
    return batch_iterative_chains


def generate_gt_tree_batch(
    original_question: str, 
    ground_truth_text: str, 
    branch_num: int, 
    llm: Any, 
    utils: Any,
    choice_text: Optional[str],
    passage: Optional[str],
    batch_size: int = 32
) -> ds.GT_Tree:

    logging.info("Starting generate gt_tree for question: %s", original_question)
    root = ds.GT_Tree(original_question=original_question, ground_truth=ground_truth_text)
    paraphrased_questions = paraphrase_questions(original_question, branch_num, llm, utils)
    total = len(paraphrased_questions)
    logging.info("Total paraphrased questions: %d", total)
    samples_remaining = total
    offset = 0
    batch_num = 1
    while samples_remaining > 0:
        n = min(batch_size, samples_remaining)
        logging.info("Processing batch %d: questions %d to %d", batch_num, offset, offset + n)
        batch_questions = paraphrased_questions[offset : offset + n]
        batch_chains = generate_batch_chain(
            questions=batch_questions, 
            ground_truth=ground_truth_text, 
            llm=llm, utils=utils, 
            choice_text=choice_text, passage=passage
        )
        root.chains += batch_chains
        offset += n
        samples_remaining -= n
        batch_num += 1
    logging.info("Finished generate_gt_tree_batch with %d chains", len(root.chains))
    return root


def generate_estimator_tree(
    original_question: str, 
    ground_truth_text: str,
    sampling_num: int, 
    branch_num: int, 
    llm: Any, 
    utils: Any,
    choice_text: Optional[str], 
    passage: Optional[str],
    max_return_per_call: int = 32
) -> ds.Estimator_Tree:

    logging.info("Starting generate estimator_tree for question: %s", original_question)
    root = ds.Estimator_Tree(original_question=original_question, ground_truth=ground_truth_text)
    # Generate paraphrased questions
    if branch_num > 1:
        paraphrased_questions = paraphrase_questions(original_question, branch_num, llm, utils)
    else:
        paraphrased_questions = [original_question]

    for idx, question in enumerate(paraphrased_questions):
        logging.info("Processing estimator subtree for paraphrased question %d: %s", idx + 1, question)
        samples_remaining = sampling_num
        answernodes = []
        batch_num = 1
        while samples_remaining > 0:
            n = min(max_return_per_call, samples_remaining)
            if utils.name in general_utils.MCQ_UTILS:
                prompt = utils.sampling_prompt.format(q=question, c=choice_text)
            elif utils.name in general_utils.RCQ_UTILS:
                prompt = utils.sampling_prompt.format(q=question, p=passage)
            else:
                prompt = utils.sampling_prompt.format(q=question)

            response = llm.generate(prompt=prompt, num_return_sequences=n)
            for text, prob in zip(response.text, response.log_prob):
                answernodes.append(
                    ds.AnswerNode(
                        raw_answer=text,
                        extract_answer=utils.extract_answer(text),
                        is_correct=utils.eval_answer(text, root.ground_truth, question),
                        token_probs=prob
                    )
                )
            logging.info("Estimator subtree for question '%s': processed batch %d with %d answers", question, batch_num, n)
            samples_remaining -= n
            batch_num += 1
        
        logging.info("< Subtree >\n paraphrased-Q: %s\n sampling-A: %s", question, [a.raw_answer for a in answernodes])
        root.subtrees.append(
            ds.Subtree(
                paraphrased_question=ds.QuestionNode(question=question),
                sampling_answers=answernodes
            )
        )    
    logging.info("Finished generate_estimator_tree with %d subtrees", len(root.subtrees))
    return root


def generate_ipt_tree(
    original_question: str, 
    ground_truth_text: str, 
    branch_num: int,
    depth: int, 
    llm: Any, 
    utils: Any,
    choice_text: Optional[str],
    passage: Optional[str],
    batch_size: int = 32
) -> ds.MI_IPT_Tree:

    logging.info("Starting generate ipt_tree for question: %s", original_question)
    root = ds.MI_IPT_Tree(original_question=original_question, ground_truth=ground_truth_text)
    paraphrased_questions = [original_question]*branch_num
    samples_remaining = len(paraphrased_questions)
    offset = 0
    batch_num = 1
    while samples_remaining > 0:
        n = min(batch_size, samples_remaining)
        logging.info("Processing batch %d: questions %d to %d", batch_num, offset, offset + n)
        batch_questions = paraphrased_questions[offset : offset + n]
        batch_chains = generate_batch_iterative_chain(
            questions=batch_questions, 
            ground_truth=ground_truth_text, round=depth,
            llm=llm, utils=utils, 
            choice_text=choice_text, passage=passage
        )
        root.chains += batch_chains
        offset += n
        samples_remaining -= n
        batch_num += 1
    logging.info("Finished generate ipt_tree with %d chains", len(root.chains))
    return root


def generate_au_tree(
    original_question: str, 
    ground_truth_text: str, 
    branch_num: int, 
    llm: Any, 
    utils: Any,
    choice_text: Optional[str],
    passage: Optional[str],
    batch_size: int = 32
) -> ds.GT_Tree:

    logging.info("Starting generate gt_tree for question: %s", original_question)
    root = ds.GT_Tree(original_question=original_question, ground_truth=ground_truth_text)
    paraphrased_questions = [original_question]*branch_num
    total = len(paraphrased_questions)
    logging.info("Total paraphrased questions: %d", total)
    samples_remaining = total
    offset = 0
    batch_num = 1
    while samples_remaining > 0:
        n = min(batch_size, samples_remaining)
        logging.info("Processing batch %d: questions %d to %d", batch_num, offset, offset + n)
        batch_questions = paraphrased_questions[offset : offset + n]
        batch_chains = generate_au_test_chain(
            questions=batch_questions, 
            ground_truth=ground_truth_text, 
            llm=llm, utils=utils, 
            choice_text=choice_text, passage=passage
        )
        root.chains += batch_chains
        offset += n
        samples_remaining -= n
        batch_num += 1
    logging.info("Finished generate au tree with %d chains", len(root.chains))
    return root


def generate_su_tree(
    original_question: str, 
    ground_truth_text: str, 
    branch_num: int, 
    llm: Any, 
    utils: Any,
    choice_text: Optional[str],
    passage: Optional[str],
    batch_size: int = 32
) -> ds.GT_Tree:

    logging.info("Starting generate gt_tree for question: %s", original_question)
    root = ds.GT_Tree(original_question=original_question, ground_truth=ground_truth_text)
    paraphrased_questions = [original_question]*branch_num
    total = len(paraphrased_questions)
    logging.info("Total paraphrased questions: %d", total)
    samples_remaining = total
    offset = 0
    batch_num = 1
    while samples_remaining > 0:
        n = min(batch_size, samples_remaining)
        logging.info("Processing batch %d: questions %d to %d", batch_num, offset, offset + n)
        batch_questions = paraphrased_questions[offset : offset + n]
        batch_chains = generate_su_test_chain(
            questions=batch_questions, 
            ground_truth=ground_truth_text, 
            llm=llm, utils=utils, 
            choice_text=choice_text, passage=passage
        )
        root.chains += batch_chains
        offset += n
        samples_remaining -= n
        batch_num += 1
    logging.info("Finished generate au tree with %d chains", len(root.chains))
    return root


def generate_eu_ou_tree(
    original_question: str, 
    ground_truth_text: str, 
    branch_num: int,
    llm: Any, 
    utils: Any,
    choice_text: Optional[str],
    passage: Optional[str],
    batch_size: int = 32
) -> ds.MI_IPT_Tree:

    logging.info("Starting generate ipt_tree for question: %s", original_question)
    root = ds.MI_IPT_Tree(original_question=original_question, ground_truth=ground_truth_text)
    paraphrased_questions = [original_question]*branch_num
    samples_remaining = len(paraphrased_questions)
    offset = 0
    batch_num = 1
    while samples_remaining > 0:
        n = min(batch_size, samples_remaining)
        logging.info("Processing batch %d: questions %d to %d", batch_num, offset, offset + n)
        batch_questions = paraphrased_questions[offset : offset + n]
        batch_chains = generate_batch_eu_ou_test_chain(
            questions=batch_questions, 
            ground_truth=ground_truth_text,
            llm=llm, utils=utils, 
            choice_text=choice_text, passage=passage
        )
        root.chains += batch_chains
        offset += n
        samples_remaining -= n
        batch_num += 1
    logging.info("Finished generate ipt_tree with %d chains", len(root.chains))
    return root




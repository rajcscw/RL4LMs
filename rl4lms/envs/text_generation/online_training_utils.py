from tqdm import tqdm
from dataclasses import dataclass
from rl4lms.envs.text_generation.training_utils import OnPolicyTrainer
from rl4lms.data_pools.text_generation_pool import Sample
from rl4lms.envs.text_generation.reward import RewardFunction
from rl4lms.envs.text_generation.evaluation_utils import generate_text
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from abc import abstractclassmethod


@dataclass
class Feedback:
    """
    Dataclass which holds expert feedback
    Two types of feedback: (1) expert demonstration (2) a scalar feedback for the generated text

    """
    ground_truth_text: str # expert provided text
    generated_text: str    # generated text by the model
    score: float           # feedback provided for the generated text 


class Expert:
    """
    Abstract class that simulates different experts with different feedback 
    """
    @abstractclassmethod
    def provide_feedback(self, sample: Sample, generated_text: str) -> Feedback:
        """
        Provides feedback for the given sample and the model generated text

        Args:
            sample (Sample): sample containing input prompt text
            generated_text (str): model generated text using current policy
        """
        raise NotImplementedError


class RewardModel(RewardFunction):
    """
    Abstract class for reward models
    """
    @abstractclassmethod
    def save(self, sample: Sample, feedback: Feedback):
        """
        Saves the expert provided feedback for further training of reward model

        Args:
            sample (Sample): input sample
            feedback (Feedback): expert feedback

        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError

    @abstractclassmethod
    def learn(self, n_steps: int):
        """
        Updates the RM with n_steps provided
        """
        raise NotImplementedError


class OnlineOnPolicyTrainer(OnPolicyTrainer):
    """
    Online trainer class
    """
    def _generate(self, sample: Sample) -> str:
        generated_texts = generate_text(self._alg.policy, 
                                        self._tokenizer,
                                        [sample],
                                        self._max_prompt_length,
                                        "",
                                        self._eval_gen_kwargs)
        return generated_texts[0]


    def train_and_eval(self):
        """
        Online training loop with simulated human in the loop
        """
        # evaluate on val and test set before fine-tuning once
        iter_start = self._trainer_state["current_iter"]
        self._evaluate_on_datapools(epoch=iter_start)


        # now we loop over training set only one (probably need to batch this too?)
        for epoch_ix, sample in tqdm(enumerate(self._samples_by_split["train"]), desc="Online loop"):
            self._trainer_state["current_iter"] = epoch_ix

            # run the sample through the model and get the predicted text
            predicted_text = self._generate(self._alg.policy, sample)

            # get the user feedback or get the expert demonstration
            feedback = self._expert.provide_feedback(sample, predicted_text)

            # now add the feedback to the reward function
            self._reward_fn.save(sample, feedback)

            # inner rollout using the learned reward function 
            # and learn loop for on-policy algorithm
            self._alg.learn(self._n_steps_per_iter)

            # save the policy checkpoint
            if (epoch_ix + 1) % self._train_eval_config.get("save_every", 20) == 0:
                self.save_trainer_state(
                    self._tracker, self._alg.policy, self._trainer_state)

            # evaluate on val set in the given intervals
            if (epoch_ix + 1) % self._train_eval_config["eval_every"] == 0:
                self._evaluate_on_datapools(epoch=epoch_ix, splits=["val"])

        # finally evaluate on val and test samples
        self._evaluate_on_datapools(epoch=epoch_ix)

        # save model here - we save only the language model
        if self._tracker is not None:
            self._tracker.save_auto_model(
                self._alg.policy.get_language_model())
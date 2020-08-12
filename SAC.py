import tensorflow as tf

from tensorflow.keras.mixed_precision import experimental as mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)

from tensorflow.keras.optimizers import Adam
import tensorflow.keras.layers as tfkl

from tensorflow_probability import distributions as tfd
import numpy as np


from tools.Module import Module
from tools.Replay_Buffer import Replay_Buffer

class SAC(Module):
	def __init__(self, action_size, obvs_size, latent_size=512, mem_size=10000):
		super(SAC, self).__init__()
		self.action_size = action_size
		self.latent_size = latent_size
		self.lr			= 0.0005
		self.counter	= 1
		self.delay		= 2
		self.batchsize	= 32
		self.tau		= 0.005
		self.gamma		= 0.99
		self.grad_clip	= 0.5
		self.target_ent	= float(-action_size)
		self.target_ent	= tf.Variable(self.target_ent, trainable=False)
		self.alpha		= tf.Variable(0.2, trainable=False)
		self.log_alpha	= tf.Variable(tf.math.log(self.alpha), trainable=True)
		self.buffer = {
				'states': np.zeros((mem_size,)+obvs_size, dtype=np.float32),
				'actions': np.zeros((mem_size, action_size), dtype=np.float32),
				'rewards': np.zeros((mem_size,), dtype=np.float32),
				'next_states': np.zeros((mem_size,)+obvs_size, dtype=np.float32),
				'dones': np.zeros((mem_size,), dtype=np.float32),
		}
		self.replay_buffer = Replay_Buffer(self.buffer, capacity=mem_size)

		self.agent   = Agent(action_size=self.action_size)
		self.critic1 = Critic(action_size=self.action_size)
		self.critic2 = Critic(action_size=self.action_size)
		self.target_agent   = Agent(action_size=self.action_size)
		self.target_critic1 = Critic(action_size=self.action_size)
		self.target_critic2 = Critic(action_size=self.action_size)

		self.agent_optimizer    = mixed_precision.LossScaleOptimizer(Adam(self.lr, epsilon=1e-8), loss_scale='dynamic')
		self.critic1_optimizer	= mixed_precision.LossScaleOptimizer(Adam(self.lr, epsilon=1e-8), loss_scale='dynamic')
		self.critic2_optimizer	= mixed_precision.LossScaleOptimizer(Adam(self.lr, epsilon=1e-8), loss_scale='dynamic')
		self.alpha_optimizer    = mixed_precision.LossScaleOptimizer(Adam(self.lr, epsilon=1e-8), loss_scale='dynamic')


	def add(self, state, action, reward, next_state, done):
		if self.counter==1:
			self.init_weights(state, action)
		self.replay_buffer.add(state, action, reward, next_state, float(not done))
		self.counter+=1


	def action(self, state):
		_, action, _ = self.agent(np.array([state]))
		return tf.squeeze(action).numpy()


	def init_weights(self, state, action):

		self.agent(np.array([state]))
		self.critic1(np.array([state]), np.array([action]))
		self.critic2(np.array([state]), np.array([action]))

		self.target_agent(np.array([state]))
		self.target_critic1(np.array([state]), np.array([action]))
		self.target_critic2(np.array([state]), np.array([action]))

		self.target_agent.set_weights(self.agent.get_weights())
		self.target_critic1.set_weights(self.critic1.get_weights())
		self.target_critic2.set_weights(self.critic2.get_weights())
		

	def update_weights(self, delay=False):
		if delay:
			self.target_agent.set_weights(list( map(lambda net, terget_net: ((self.tau) * net) + ((1-self.tau) * terget_net), self.agent.get_weights(), self.target_agent.get_weights()) ))
			self.alpha.assign(tf.exp(self.log_alpha))
		self.target_critic1.set_weights(list( map(lambda net, target_net: ((self.tau) * net) + ((1-self.tau) * target_net), self.critic1.get_weights(), self.target_critic1.get_weights()) ))
		self.target_critic2.set_weights(list( map(lambda net, target_net: ((self.tau) * net) + ((1-self.tau) * target_net), self.critic2.get_weights(), self.target_critic2.get_weights()) ))


	def update(self):
		if self.replay_buffer.is_ready(self.batchsize):
			delay = bool(self.counter%self.delay)
			states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batchsize)

			self.critic_update(states, actions, rewards, next_states, dones)
			if delay:
				self.agent_update(states)

			self.update_weights(delay=delay)
		


	@tf.function
	def agent_update(self, states):
		with tf.GradientTape() as actor_tape, tf.GradientTape() as alpha_tape:
			_, actions, entropy = self.agent(states)
			_, q_value1 = self.critic1(states, actions)
			_, q_value2 = self.critic2(states, actions)
			q_value     = tf.minimum(q_value1, q_value2)
			agent_loss  = tf.reduce_mean(-q_value + tf.stop_gradient(self.alpha) * -entropy)
			alpha_loss  = tf.reduce_mean(self.log_alpha * tf.stop_gradient(entropy - self.target_ent))

			agent_loss	= self.agent_optimizer.get_scaled_loss(agent_loss)
			# alpha_loss	= self.alpha_optimizer.get_scaled_loss(alpha_loss)

		agent_grads = actor_tape.gradient(agent_loss, self.agent.trainable_variables)
		alpha_grads = alpha_tape.gradient(alpha_loss, [self.log_alpha])

		agent_grads = self.agent_optimizer.get_unscaled_gradients(agent_grads)
		# alpha_grads = self.alpha_optimizer.get_unscaled_gradients(alpha_grads)

		# agent_grads, _ = tf.clip_by_global_norm(agent_grads, self.grad_clip)
		# alpha_grads, _ = tf.clip_by_global_norm(alpha_grads, self.grad_clip)

		self.agent_optimizer.apply_gradients(zip(agent_grads, self.agent.trainable_variables))
		self.alpha_optimizer.apply_gradients(zip(alpha_grads, [self.log_alpha]))

		return agent_loss, alpha_loss


	@tf.function
	def critic_update(self, states, actions, rewards, next_states, dones):
		with tf.GradientTape() as tape_1, tf.GradientTape() as tape_2:
			rewards     = tf.expand_dims(rewards, axis=-1)
			dones       = tf.expand_dims(dones, axis=-1)
			q_dist1, _  = self.critic1(states, actions)
			q_dist2, _  = self.critic2(states, actions)
			
			_, next_actions, next_entropy = self.target_agent(next_states)

			_, next_q_value1	= self.target_critic1(next_states, next_actions)
			_, next_q_value2	= self.target_critic2(next_states, next_actions)
			next_q_value			= tf.minimum(next_q_value1, next_q_value2)
			next_q_value			= next_q_value + self.alpha * next_entropy

			target_q    = tf.stop_gradient(rewards + self.gamma * next_q_value * dones)
			loss_1      = -tf.reduce_mean(q_dist1.log_prob(target_q))
			loss_2      = -tf.reduce_mean(q_dist2.log_prob(target_q))

			loss_1	= self.critic1_optimizer.get_scaled_loss(loss_1)
			loss_2	= self.critic2_optimizer.get_scaled_loss(loss_2)

		grads_1 = tape_1.gradient(loss_1, self.critic1.trainable_variables)
		grads_2 = tape_2.gradient(loss_2, self.critic2.trainable_variables)

		grads_1 = self.critic1_optimizer.get_unscaled_gradients(grads_1)
		grads_2 = self.critic2_optimizer.get_unscaled_gradients(grads_2)

		# grads_1, _ = tf.clip_by_global_norm(grads_1, self.grad_clip)
		# grads_2, _ = tf.clip_by_global_norm(grads_2, self.grad_clip)

		self.critic1_optimizer.apply_gradients(zip(grads_1, self.critic1.trainable_variables))
		self.critic2_optimizer.apply_gradients(zip(grads_2, self.critic2.trainable_variables))

		return loss_1, loss_2


class Agent(Module):
	def __init__(self, action_size):
		super(Agent, self).__init__()
		self.action_size = action_size

	def __call__(self, s):
		if len(s.shape)==4:
			kwargs = dict(padding='valid', activation='swish', kernel_initializer='lecun_normal')
			h = self.get('vision_1', tfkl.Conv2D, 32, 8, 4, **kwargs)(s)
			h = self.get('vision_2', tfkl.Conv2D, 64, 4, 2, **kwargs)(h)
			h = self.get('vision_3', tfkl.Conv2D, 64, 3, 1, **kwargs)(h)
			# h = self.get('vision_4', tfkl.Conv2D, 64, 3, 2, **kwargs)(h)
			s = tfkl.Flatten()(h)

		h = self.get('agent_1', tfkl.Dense, 512, activation='swish', kernel_initializer='lecun_normal')(s)
		h = self.get('agent_2', tfkl.Dense, 512, activation='swish', kernel_initializer='lecun_normal')(h)
		# h = self.get('agent_3', tfkl.Dense, 512, activation='swish', kernel_initializer='lecun_normal')(h)
		o = self.get('agent_4', tfkl.Dense, 2*self.action_size, activation='linear', kernel_initializer='lecun_normal')(h)
		o = tf.cast(o, tf.float32)

		mean, std   = tf.split(o, 2, -1)
		std         = tf.nn.softplus(std) + 1.0
		dist        = tfd.Normal(mean, std, )
		logit		= dist.sample()
		action		= tf.nn.tanh(logit)
		entropy		= dist.log_prob(logit)
		entropy		= -tf.reduce_sum(entropy - tf.math.log(1 - (action ** 2) + 1e-10), -1, keepdims=True) # The Entropy

		return dist, action, entropy


class Critic(Module):
	def __init__(self, action_size):
		super(Critic, self).__init__()
		self.action_size = action_size

	def __call__(self, s, a):
		if len(s.shape)==4:
			kwargs = dict(padding='valid', activation='swish', kernel_initializer='lecun_normal')
			h = self.get('vision_1', tfkl.Conv2D, 32, 8, 4, **kwargs)(s)
			h = self.get('vision_2', tfkl.Conv2D, 64, 4, 2, **kwargs)(h)
			h = self.get('vision_3', tfkl.Conv2D, 64, 3, 1, **kwargs)(h)
			# h = self.get('vision_4', tfkl.Conv2D, 64, 3, 2, **kwargs)(h)
			s = tfkl.Flatten()(h)

		s = self.get('value_1', tfkl.Dense, 512, activation='swish', kernel_initializer='lecun_normal')(s)
		a = self.get('value_2', tfkl.Dense, 512, activation='swish', kernel_initializer='lecun_normal')(a)
		h = tf.add(s, a)
		# h = tf.concat([s, a], -1)
		# h = self.get('value_3', tfkl.Dense, 512, activation='swish', kernel_initializer='lecun_normal')(h)
		h = self.get('value_4', tfkl.Dense, 512, activation='swish', kernel_initializer='lecun_normal')(h)
		q = self.get('value_5', tfkl.Dense, 1, activation='linear', kernel_initializer='lecun_normal')(h)
		q = tf.cast(q, tf.float32)

		dist	= tfd.Normal(q, 1)
		value	= dist.sample()

		return dist, value

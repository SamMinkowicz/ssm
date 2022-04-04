import autograd.numpy as np
import autograd.numpy.random as npr
npr.seed(0)

import matplotlib.pyplot as plt

import ssm
from ssm.util import find_permutation

# Set the parameters of the HMM
T = 500     # number of time bins
K = 5       # number of discrete states
D = 2       # number of observed dimensions

# Make an HMM with the true parameters
true_hmm = ssm.HMM(K, D, observations="ar")
z_train, y_train = true_hmm.sample(T)
z_test, y_test = true_hmm.sample(T)
true_ll = true_hmm.log_probability(y_train)

# Fit models
N_sgd_iters = 1000
N_em_iters = 100

# Specify observation model
observations = [
    "ar"
]

# Fit with both SGD and EM
methods = ["sgd", "em"]

results = {}
for obs in observations:
    for method in methods:
        print("Fitting {} HMM with {}".format(obs, method))
        model = ssm.HMM(K, D, observations=obs)
        train_lls = model.fit(y_train, method=method)
        test_ll = model.log_likelihood(y_test)
        smoothed_y = model.smooth(y_train)

        # Permute to match the true states
        model.permute(find_permutation(z_train, model.most_likely_states(y_train)))
        smoothed_z = model.most_likely_states(y_train)
        results[(obs, method)] = (model, train_lls, test_ll, smoothed_z, smoothed_y)

# Plot the inferred states
fig, axs = plt.subplots(len(observations) + 1, 1, figsize=(12, 8))

# Plot the true states
plt.sca(axs[0])
plt.imshow(z_train[None, :], aspect="auto", cmap="jet")
plt.title("true")
plt.xticks()

# Plot the inferred states
for i, obs in enumerate(observations):
    zs = []
    for method, ls in zip(methods, ['-', ':']):
        _, _, _, smoothed_z, _ = results[(obs, method)]
        zs.append(smoothed_z)

    plt.sca(axs[i+1])
    plt.imshow(np.row_stack(zs), aspect="auto", cmap="jet")
    plt.yticks([0, 1], methods)
    if i != len(observations) - 1:
        plt.xticks()
    else:
        plt.xlabel("time")
    plt.title(obs)

plt.tight_layout()

# Plot smoothed observations
fig, axs = plt.subplots(D, 1, figsize=(12, 8))

# Plot the true data
for d in range(D):
    plt.sca(axs[d])
    plt.plot(y_train[:, d], '-k', lw=2, label="True")
    plt.xlabel("time")
    plt.ylabel("$y_{{}}$".format(d+1))

for obs in observations:
    line = None
    for method, ls in zip(methods, ['-', ':']):
        _, _, _, _, smoothed_y = results[(obs, method)]
        for d in range(D):
            plt.sca(axs[d])
            color = line.get_color() if line is not None else None
            line = plt.plot(smoothed_y[:, d], ls=ls, lw=1, color=color, label="{}({})".format(obs, method))[0]

# Make a legend
plt.sca(axs[0])
plt.legend(loc="upper right")
plt.tight_layout()

# Plot log likelihoods
plt.figure(figsize=(12, 8))
for obs in observations:
    line = None
    for method, ls in zip(methods, ['-', ':']):
        _, lls, _, _, _ = results[(obs, method)]
        color = line.get_color() if line is not None else None
        line = plt.plot(lls, ls=ls, lw=1, color=color, label="{}({})".format(obs, method))[0]

xlim = plt.xlim()
plt.plot(xlim, true_ll * np.ones(2), '-k', label="true")
plt.xlim(xlim)

plt.legend(loc="lower right")
plt.tight_layout()

# Print the test log likelihoods
print("Test log likelihood")
print("True: ", true_hmm.log_likelihood(y_test))
for obs in observations:
    for method in methods:
        _, _, test_ll, _, _ = results[(obs, method)]
        print("{} ({}): {}".format(obs, method, test_ll))

plt.show()

# Plot the transition matrices
fig, axs = plt.subplots(1, len(observations)*len(methods) + 1, figsize=(12, 8))

# Plot the true transition matrix
true_transition_mat = true_hmm.transitions.transition_matrix
plt.sca(axs[0])
im = plt.imshow(true_transition_mat, cmap='gray')
plt.title("True Transition Matrix")

# plot the inferred transition matrix
i = 1
for obs in observations:
    for method in methods:
        model, _, _, _, _ = results[(obs, method)]

        plt.sca(axs[i])
        learned_transition_mat = model.transitions.transition_matrix
        im = plt.imshow(learned_transition_mat, cmap='gray')
        plt.title(f"Learned Transition Matrix - {method}")
        i += 1

plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
cbar_ax = fig.add_axes([0.85, 0.1, 0.05, 0.7])
fig.colorbar(im, cax=cbar_ax)
plt.show()

# Plot state distributions
fig, axs = plt.subplots(len(observations)*len(methods) + 1, 1, figsize=(12, 8), sharex=True)
true_state_list, true_durations = ssm.util.rle(z_train)

# Rearrange the lists of durations to be a nested list where
# the nth inner list is a list of durations for state n
true_durs_stacked = [true_durations[true_state_list == s] for s in range(K)]

plt.sca(axs[0])
plt.hist(true_durs_stacked, label=['state ' + str(s) for s in range(K)])
plt.ylabel('Frequency')
plt.legend()
plt.title('Histogram of True State Durations')

i = 1
for obs in observations:
    for method in methods:
        _, _, _, smoothed_z, _ = results[(obs, method)]

        inferred_state_list, inferred_durations = ssm.util.rle(smoothed_z)
        inf_durs_stacked = [inferred_durations[inferred_state_list == s] for s in range(K)]

        plt.sca(axs[i])
        plt.hist(inf_durs_stacked, label=['state ' + str(s) for s in range(K)])
        plt.ylabel('Frequency')
        plt.legend()
        plt.title(f'Histogram of Inferred State Durations - {method}')
        i += 1

plt.xlabel('Duration')
plt.show()

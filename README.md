# On decentralized computation of the leader's strategy in bi-level games

Arxiv preprint: https://arxiv.org/abs/2402.14449, 2024

**Abstract:** Motivated by the omnipresence of hierarchical structures in many real-world applications, this study delves into the intricate
realm of bi-level games, with a specific focus on exploring local Stackelberg equilibria as a solution concept. While existing
literature offers various methods tailored to specific game structures featuring one leader and multiple followers, a compre-
hensive framework providing formal convergence guarantees to a local Stackelberg equilibrium appears to be lacking. Drawing
inspiration from sensitivity results for nonlinear programs and guided by the imperative to maintain scalability and preserve
agent privacy, we propose a decentralized approach based on the projected gradient descent with the Armijo stepsize rule.
The main challenge here lies in assuring the existence and well-posedness of Jacobians that describe the leader’s decision’s
influence on the achieved equilibrium of the followers. By meticulous tracking of the Implicit Function Theorem requirements
at each iteration, we establish formal convergence guarantees to a local Stackelberg equilibrium for a broad class of bi-level
games. Building on our prior work on quadratic aggregative Stackelberg games, we also introduce a decentralized warm-start
procedure based on the consensus alternating direction method of multipliers addressing the previously reported initialization
issues. Finally, we provide empirical validation through two case studies in smart mobility, showcasing the effectiveness of our
general method in handling general convex constraints, and the effectiveness of its extension in tackling initialization issues.

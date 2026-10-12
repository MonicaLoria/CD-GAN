CD-GAN: Correlation-Preserving Dual-Head Discriminator GAN for Realistic Tabular Data Synthesis.

CD-GAN is a correlation-preserving generative adversarial network designed for realistic synthetic tabular data generation. The model provides complementary adversarial supervision at two levels:
- a row-level branch that evaluates the realism and coherence of complete encoded records;
- a feature-level branch composed of separate column-specific critics that evaluate individual original predictor attributes.
For numerical attributes, the standardized scalar representation is evaluated. For categorical attributes, the complete one-hot encoded block is treated as a single original feature. This prevents high-cardinality categorical variables from receiving greater feature-level importance simply because they occupy more encoded dimensions.
CD-GAN further includes a covariance-based dependency regularization term that encourages preservation of second-order relationships between real and synthetic data in the encoded space. Wasserstein adversarial training with gradient penalty is used to improve training stability.

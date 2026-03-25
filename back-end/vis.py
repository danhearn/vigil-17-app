        fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

        label_fontsize = 16  # size for axis labels
        title_fontsize = 18  # size for titles
        tick_fontsize = 16   # size for tick labels

        axes[0].plot(x_arr, sine_n, color="steelblue", linewidth=1.2)
        axes[0].set_ylabel("Amplitude", fontsize=label_fontsize)
        axes[0].set_title("Base Sine Wave", fontsize=title_fontsize)
        axes[0].set_ylim(-1.1, 1.1)
        axes[0].axhline(0, color="white", linewidth=0.5, alpha=0.3)

        axes[1].plot(x_arr, temporal_n, color="tomato", linewidth=1.2)
        axes[1].set_ylabel("Amplitude", fontsize=label_fontsize)
        axes[1].set_title("Depth Contour Signal", fontsize=title_fontsize)
        axes[1].set_ylim(-1.1, 1.1)
        axes[1].axhline(0, color="white", linewidth=0.5, alpha=0.3)

        axes[2].plot(x_arr, displaced_n, color="mediumpurple", linewidth=1.2)
        axes[2].fill_between(
            x_arr, displaced_n, np.zeros_like(displaced_n),
            alpha=0.15, color="mediumpurple"
        )
        axes[2].set_ylabel("Amplitude", fontsize=label_fontsize)
        axes[2].set_title("Wavetable", fontsize=title_fontsize)
        axes[2].set_ylim(-1.1, 1.1)
        axes[2].axhline(0, color="white", linewidth=0.5, alpha=0.3)

        for ax in axes:
            ax.set_xticklabels([])       # remove x-axis numbers
            ax.tick_params(axis='x', which='both', bottom=True)  # keep ticks if you want

        plt.tight_layout()
        plt.savefig(f"wavetable_superposition/sine_displaced_frame_{i+1:06d}.png", dpi=120)
        plt.close(fig)
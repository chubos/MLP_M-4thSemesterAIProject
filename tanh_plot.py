import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

output_dir = Path('figures')
output_dir.mkdir(exist_ok=True)

x = np.linspace(-5, 5, 1000)
y = np.tanh(x)

plt.figure(figsize=(7, 4.5))
plt.plot(x, y, color='#0B4F6C', linewidth=2.5, label=r'$\tanh(z)$')
plt.axhline(0, color='black', linewidth=0.8)
plt.axvline(0, color='black', linewidth=0.8)
plt.axhline(1, color='#999999', linestyle='--', linewidth=1)
plt.axhline(-1, color='#999999', linestyle='--', linewidth=1)
plt.xlim(-5, 5)
plt.ylim(-1.1, 1.1)
plt.xticks([-5, -3, -1, 0, 1, 3, 5])
plt.yticks([-1, -0.5, 0, 0.5, 1])
plt.grid(True, linestyle='--', alpha=0.35)
plt.xlabel('z')
plt.ylabel('h(z)')
plt.legend(frameon=False, loc='lower right')
plt.tight_layout()
plt.savefig(output_dir / 'tanh.png', dpi=300, bbox_inches='tight')
plt.close()

print('Saved figures/tanh.png')

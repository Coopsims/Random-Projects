import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc
import os

def get_available_filename(filename):
    """
    Check if a file exists and return a new filename with an incremental suffix if it does.
    
    Args:
        filename (str): The original filename to check
        
    Returns:
        str: An available filename that doesn't exist yet
    """
    if not os.path.exists(filename):
        return filename
        
    # Split the filename into name and extension
    name, ext = os.path.splitext(filename)
    
    # Try with incremental suffixes (_1, _2, etc.) until we find an available name
    counter = 1
    while True:
        new_filename = f"{name}_{counter}{ext}"
        if not os.path.exists(new_filename):
            return new_filename
        counter += 1

def recaman_sequence(n):
    """
    Calculate the Recamán sequence up to index n.
    
    Args:
        n (int): The number of elements to generate in the sequence
        
    Returns:
        list: The Recamán sequence from a(0) to a(n)
    """
    sequence = [0]  # Start with a(0) = 0
    seen = set([0])  # Keep track of numbers we've seen
    
    for i in range(1, n + 1):
        # Try to go backwards (a(n-1) - n)
        prev = sequence[-1]
        next_val = prev - i
        
        # If next_val is positive and not already in the sequence, use it
        # Otherwise, go forward (a(n-1) + n)
        if next_val > 0 and next_val not in seen:
            sequence.append(next_val)
        else:
            next_val = prev + i
            sequence.append(next_val)
            
        seen.add(next_val)
        
    return sequence

def plot_recaman(sequence, save_path=None):
    """
    Plot the Recamán sequence as half circles alternating above and below a horizontal line.
    
    Args:
        sequence (list): The Recamán sequence to plot
        save_path (str, optional): Path to save the plot. If None, the plot is displayed.
    """
    # Scale figure size based on the size of n to prevent muddled lines
    # For larger sequences, we need a larger figure with better scaling
    n = len(sequence) - 1
    
    # More aggressive scaling for very large n values
    if n > 1000:
        width = max(12, min(36, 12 + (n // 50)))
        height = max(6, min(18, 6 + (n // 100)))
    else:
        width = max(12, min(24, 12 + (n // 100)))
        height = max(6, min(12, 6 + (n // 100)))
    
    fig, ax = plt.subplots(figsize=(width, height))
    
    # Set up the plot
    ax.set_xlim(0, max(sequence) + 1)
    
    # Adjust y-axis limits based on n for better visualization
    max_val = max(sequence)
    if n > 1000:
        # For very large n, use a smaller vertical range to focus on the pattern
        y_limit = max_val // 6
    else:
        # For smaller n, use the original vertical range
        y_limit = max_val // 4
        
    ax.set_ylim(-y_limit, y_limit)
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Calculate line width based on n - thinner lines for higher n values
    if n > 1000:
        # Even thinner lines for very large n values
        line_width = max(0.1, min(1.0, 1.0 * (100 / max(100, n))))
    else:
        # Standard line width scaling for smaller n values
        line_width = max(0.1, min(1.5, 1.5 * (100 / max(100, n))))
    
    # Plot the sequence as half circles
    for i in range(len(sequence) - 1):
        start = min(sequence[i], sequence[i+1])
        end = max(sequence[i], sequence[i+1])
        diameter = end - start
        
        # Alternate between drawing arcs above and below the line
        if i % 2 == 0:
            # Draw arc above the line
            arc = Arc((start + diameter/2, 0), diameter, diameter, 
                     theta1=0, theta2=180, color='Black', lw=line_width)
        else:
            # Draw arc below the line
            arc = Arc((start + diameter/2, 0), diameter, diameter, 
                     theta1=180, theta2=360, color='Black', lw=line_width)
        
        ax.add_patch(arc)
    
    # Remove axis ticks and labels for a cleaner look
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    
    # Remove spines (borders)
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        # Get an available filename to avoid overwriting existing files
        available_path = get_available_filename(save_path)
        
        # Scale DPI based on n - higher resolution for larger n values
        if n > 1000:
            # More aggressive DPI scaling for very large n values
            dpi = min(2000, 300 + (n // 50) * 30)
        else:
            # Standard DPI scaling for smaller n values
            dpi = 1600
        
        plt.savefig(available_path, dpi=dpi, bbox_inches='tight')
        print(f"Plot saved to {available_path} with DPI={dpi}")
    else:
        plt.show()

def main():
    # Generate the Recamán sequence up to n=2000
    n = 80
    sequence = recaman_sequence(n)
    
    print(f"Recamán sequence up to n={n}:")
    print(sequence)
    # Plot the sequence and save to file
    plot_recaman(sequence, save_path="recaman_visualization.png")
    
    # You can also display the plot interactively by uncommenting the line below
    # plot_recaman(sequence)

if __name__ == "__main__":
    main()
import numpy as np
from scipy import signal, ndimage
from scipy.fft import fft2, ifft2, fftfreq
from skimage import filters, measure, exposure
import cv2

class MultiCandCombination:
    """
    Multi-Band Combination module for high-fidelity image reconstruction
    Aggregates complementary frequency bands to reconstruct palm images
    """
    
    def __init__(self, combination_method='weighted_average', 
                 weight_scheme='snr', fusion_mode='frequency'):
        """
        Initialize Multi-Band Combination module
        
        Args:
            combination_method: Method for combining bands 
                               ('weighted_average', 'maximum_selection', 'adaptive_fusion')
            weight_scheme: Weighting scheme for combination 
                          ('snr', 'entropy', 'edge', 'adaptive')
            fusion_mode: Domain for fusion ('frequency', 'spatial', 'hybrid')
        """
        self.combination_method = combination_method
        self.weight_scheme = weight_scheme
        self.fusion_mode = fusion_mode
        
    def compute_band_quality_metrics(self, reconstructed_image):
        """
        Compute quality metrics for a reconstructed band image
        
        Args:
            reconstructed_image: Reconstructed image from a single band
            
        Returns:
            Dictionary of quality metrics
        """
        # Signal-to-Noise Ratio estimation
        snr = self._estimate_snr(reconstructed_image)
        
        # Image entropy (information content)
        img_entropy = measure.shannon_entropy(reconstructed_image)
        
        # Edge strength (structural detail)
        edge_map = filters.sobel(reconstructed_image)
        edge_strength = np.mean(np.abs(edge_map))
        
        # Contrast measure
        contrast = np.std(reconstructed_image)
        
        # Sharpness measure (Laplacian variance)
        laplacian = cv2.Laplacian(reconstructed_image.astype(np.float32), cv2.CV_64F)
        sharpness = laplacian.var()
        
        return {
            'snr': snr,
            'entropy': img_entropy,
            'edge_strength': edge_strength,
            'contrast': contrast,
            'sharpness': sharpness
        }
    
    def _estimate_snr(self, image):
        """
        Estimate Signal-to-Noise Ratio of an image
        
        Args:
            image: Input image
            
        Returns:
            Estimated SNR value
        """
        # Use local variance method for SNR estimation
        mean = cv2.GaussianBlur(image, (5, 5), 0)
        diff = image - mean
        noise_var = np.var(diff)
        signal_var = np.var(image)
        
        if noise_var > 0:
            snr = 10 * np.log10(signal_var / noise_var)
        else:
            snr = float('inf')
            
        return snr
    
    def compute_combination_weights(self, band_images, quality_metrics_list):
        """
        Compute weights for combining multiple band images
        
        Args:
            band_images: List of reconstructed images from different bands
            quality_metrics_list: List of quality metrics for each band
            
        Returns:
            Normalized weights for each band
        """
        n_bands = len(band_images)
        weights = np.zeros(n_bands)
        
        if self.weight_scheme == 'snr':
            # Weight by Signal-to-Noise Ratio
            for i, metrics in enumerate(quality_metrics_list):
                weights[i] = metrics['snr']
                
        elif self.weight_scheme == 'entropy':
            # Weight by information content
            for i, metrics in enumerate(quality_metrics_list):
                weights[i] = metrics['entropy']
                
        elif self.weight_scheme == 'edge':
            # Weight by edge strength
            for i, metrics in enumerate(quality_metrics_list):
                weights[i] = metrics['edge_strength']
                
        elif self.weight_scheme == 'adaptive':
            # Adaptive weighting based on multiple factors
            for i, metrics in enumerate(quality_metrics_list):
                # Combine multiple metrics with adaptive weights
                weights[i] = (0.3 * self._normalize_metric(metrics['snr'], [m['snr'] for m in quality_metrics_list]) +
                             0.25 * self._normalize_metric(metrics['entropy'], [m['entropy'] for m in quality_metrics_list]) +
                             0.25 * self._normalize_metric(metrics['edge_strength'], [m['edge_strength'] for m in quality_metrics_list]) +
                             0.2 * self._normalize_metric(metrics['sharpness'], [m['sharpness'] for m in quality_metrics_list]))
        
        # Normalize weights to sum to 1
        if np.sum(weights) > 0:
            weights = weights / np.sum(weights)
        else:
            weights = np.ones(n_bands) / n_bands  # Equal weights as fallback
            
        return weights
    
    def _normalize_metric(self, value, all_values):
        """
        Normalize a metric value relative to all values
        
        Args:
            value: Single metric value
            all_values: List of all metric values
            
        Returns:
            Normalized value [0, 1]
        """
        min_val = np.min(all_values)
        max_val = np.max(all_values)
        
        if max_val - min_val > 0:
            return (value - min_val) / (max_val - min_val)
        else:
            return 0.5
    
    def spatial_domain_fusion(self, band_images, weights):
        """
        Combine band images in spatial domain
        
        Args:
            band_images: List of reconstructed images
            weights: Combination weights for each band
            
        Returns:
            Combined image
        """
        if self.combination_method == 'weighted_average':
            # Weighted average combination
            combined = np.zeros_like(band_images[0], dtype=np.float64)
            for img, w in zip(band_images, weights):
                combined += w * img
                
        elif self.combination_method == 'maximum_selection':
            # Pixel-wise maximum selection
            stacked = np.stack(band_images, axis=2)
            weight_map = np.repeat(weights.reshape(1, 1, -1), 
                                  stacked.shape[0], axis=0)
            weight_map = np.repeat(weight_map, stacked.shape[1], axis=1)
            weighted_stack = stacked * weight_map
            combined = np.max(weighted_stack, axis=2)
            
        elif self.combination_method == 'adaptive_fusion':
            # Adaptive fusion based on local quality
            combined = self._adaptive_spatial_fusion(band_images, weights)
            
        return combined
    
    def frequency_domain_fusion(self, band_images, weights):
        """
        Combine band images in frequency domain
        
        Args:
            band_images: List of reconstructed images
            weights: Combination weights for each band
            
        Returns:
            Combined image
        """
        # Transform all images to frequency domain
        freq_representations = []
        for img in band_images:
            freq_representations.append(fft2(img))
        
        # Combine in frequency domain
        combined_freq = np.zeros_like(freq_representations[0])
        
        if self.combination_method == 'weighted_average':
            # Weighted average of frequency components
            for freq, w in zip(freq_representations, weights):
                combined_freq += w * freq
                
        elif self.combination_method == 'maximum_selection':
            # Select maximum magnitude for each frequency
            for i, (freq, w) in enumerate(zip(freq_representations, weights)):
                if i == 0:
                    combined_freq = freq * w
                    max_magnitude = np.abs(freq) * w
                else:
                    current_magnitude = np.abs(freq) * w
                    mask = current_magnitude > max_magnitude
                    combined_freq[mask] = freq[mask] * w
                    max_magnitude[mask] = current_magnitude[mask]
                    
        elif self.combination_method == 'adaptive_fusion':
            # Frequency-dependent weighting
            combined_freq = self._adaptive_frequency_fusion(freq_representations, weights)
        
        # Transform back to spatial domain
        combined = np.real(ifft2(combined_freq))
        
        return combined
    
    def _adaptive_spatial_fusion(self, band_images, global_weights):
        """
        Perform adaptive spatial fusion with local quality assessment
        
        Args:
            band_images: List of reconstructed images
            global_weights: Global weights for each band
            
        Returns:
            Adaptively fused image
        """
        h, w = band_images[0].shape
        combined = np.zeros((h, w), dtype=np.float64)
        
        # Define local window size
        window_size = 16
        
        for y in range(0, h, window_size):
            for x in range(0, w, window_size):
                # Extract local patches
                y_end = min(y + window_size, h)
                x_end = min(x + window_size, w)
                
                local_patches = []
                local_qualities = []
                
                for img in band_images:
                    patch = img[y:y_end, x:x_end]
                    local_patches.append(patch)
                    
                    # Compute local quality metrics
                    local_var = np.var(patch)
                    local_edge = np.mean(np.abs(filters.sobel(patch)))
                    local_quality = local_var * local_edge
                    local_qualities.append(local_quality)
                
                # Compute local weights
                local_qualities = np.array(local_qualities)
                local_weights = local_qualities * global_weights
                
                if np.sum(local_weights) > 0:
                    local_weights = local_weights / np.sum(local_weights)
                else:
                    local_weights = global_weights
                
                # Combine patches
                combined_patch = np.zeros_like(local_patches[0])
                for patch, w in zip(local_patches, local_weights):
                    combined_patch += w * patch
                
                combined[y:y_end, x:x_end] = combined_patch
        
        return combined
    
    def _adaptive_frequency_fusion(self, freq_representations, weights):
        """
        Perform adaptive frequency domain fusion
        
        Args:
            freq_representations: List of frequency domain representations
            weights: Global weights for each band
            
        Returns:
            Combined frequency representation
        """
        combined_freq = np.zeros_like(freq_representations[0])
        h, w = freq_representations[0].shape
        
        # Create frequency-dependent weight maps
        freq_y = fftfreq(h).reshape(-1, 1)
        freq_x = fftfreq(w).reshape(1, -1)
        freq_radius = np.sqrt(freq_y**2 + freq_x**2)
        
        # Define frequency bands for adaptive weighting
        low_freq_mask = freq_radius < 0.1
        mid_freq_mask = (freq_radius >= 0.1) & (freq_radius < 0.3)
        high_freq_mask = freq_radius >= 0.3
        
        for i, (freq, w) in enumerate(zip(freq_representations, weights)):
            # Adjust weights based on frequency content
            freq_weight = np.ones_like(freq_radius) * w
            
            # Boost weights for bands with strong content in specific frequency ranges
            magnitude = np.abs(freq)
            
            # Adaptive weight adjustment based on frequency content strength
            low_strength = np.mean(magnitude[low_freq_mask])
            mid_strength = np.mean(magnitude[mid_freq_mask])
            high_strength = np.mean(magnitude[high_freq_mask])
            
            total_strength = low_strength + mid_strength + high_strength
            if total_strength > 0:
                freq_weight[low_freq_mask] *= (low_strength / total_strength)
                freq_weight[mid_freq_mask] *= (mid_strength / total_strength)
                freq_weight[high_freq_mask] *= (high_strength / total_strength)
            
            combined_freq += freq * freq_weight
        
        return combined_freq
    
    def hybrid_fusion(self, band_images, weights):
        """
        Hybrid fusion combining spatial and frequency domain approaches
        
        Args:
            band_images: List of reconstructed images
            weights: Combination weights for each band
            
        Returns:
            Combined image using hybrid approach
        """
        # Perform both spatial and frequency domain fusion
        spatial_combined = self.spatial_domain_fusion(band_images, weights)
        frequency_combined = self.frequency_domain_fusion(band_images, weights)
        
        # Combine the results with adaptive weighting
        # Use frequency domain for global structure, spatial for local details
        hybrid_combined = 0.6 * frequency_combined + 0.4 * spatial_combined
        
        return hybrid_combined
    
    def combine_bands(self, band_images, band_frequencies=None):
        """
        Main function to combine multiple frequency bands
        
        Args:
            band_images: List of reconstructed images from informative bands
            band_frequencies: Optional list of (f_low, f_high) tuples for each band
            
        Returns:
            Dictionary containing:
                - combined_image: Final reconstructed image
                - weights: Weights used for combination
                - quality_metrics: Quality metrics for each band
        """
        # Ensure all images have the same dimensions
        target_shape = band_images[0].shape
        normalized_images = []
        
        for img in band_images:
            if img.shape != target_shape:
                # Resize if necessary
                img_resized = cv2.resize(img, (target_shape[1], target_shape[0]))
                normalized_images.append(img_resized)
            else:
                normalized_images.append(img)
        
        # Compute quality metrics for each band
        quality_metrics_list = []
        for img in normalized_images:
            metrics = self.compute_band_quality_metrics(img)
            quality_metrics_list.append(metrics)
        
        # Compute combination weights
        weights = self.compute_combination_weights(normalized_images, quality_metrics_list)
        
        # Perform combination based on selected fusion mode
        if self.fusion_mode == 'spatial':
            combined_image = self.spatial_domain_fusion(normalized_images, weights)
        elif self.fusion_mode == 'frequency':
            combined_image = self.frequency_domain_fusion(normalized_images, weights)
        elif self.fusion_mode == 'hybrid':
            combined_image = self.hybrid_fusion(normalized_images, weights)
        else:
            raise ValueError(f"Unknown fusion mode: {self.fusion_mode}")
        
        # Post-processing
        combined_image = self.post_process_combined_image(combined_image)
        
        return {
            'combined_image': combined_image,
            'weights': weights,
            'quality_metrics': quality_metrics_list,
            'band_frequencies': band_frequencies
        }
    
    def post_process_combined_image(self, image):
        """
        Apply post-processing to enhance the combined image
        
        Args:
            image: Combined image
            
        Returns:
            Post-processed image
        """
        # Normalize to [0, 1] range
        img_normalized = (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-8)
        
        # Apply contrast enhancement
        img_enhanced = exposure.equalize_adapthist(img_normalized, clip_limit=0.03)
        
        # Denoise while preserving edges
        img_denoised = filters.median(img_enhanced, disk(1))
        
        # Apply mild sharpening
        kernel = np.array([[-0.5, -1, -0.5],
                           [-1, 7, -1],
                           [-0.5, -1, -0.5]]) / 2
        img_sharpened = cv2.filter2D(img_denoised.astype(np.float32), -1, kernel)
        
        # Ensure output is in valid range
        img_final = np.clip(img_sharpened, 0, 1)
        
        return img_final
    
    def analyze_combination_quality(self, combined_result):
        """
        Analyze the quality of the combined reconstruction
        
        Args:
            combined_result: Result dictionary from combine_bands
            
        Returns:
            Dictionary with quality analysis
        """
        combined_image = combined_result['combined_image']
        
        # Overall quality metrics
        overall_entropy = measure.shannon_entropy(combined_image)
        overall_sharpness = cv2.Laplacian(combined_image.astype(np.float32), cv2.CV_64F).var()
        overall_contrast = np.std(combined_image)
        
        # Edge detection for structure assessment
        edges = filters.canny(combined_image, sigma=1.0)
        edge_density = np.sum(edges) / edges.size
        
        # Frequency content analysis
        freq_spectrum = np.abs(fft2(combined_image))
        freq_energy = np.sum(freq_spectrum**2)
        
        analysis = {
            'overall_entropy': overall_entropy,
            'overall_sharpness': overall_sharpness,
            'overall_contrast': overall_contrast,
            'edge_density': edge_density,
            'frequency_energy': freq_energy,
            'combination_weights': combined_result['weights'],
            'band_contributions': self._analyze_band_contributions(combined_result)
        }
        
        return analysis
    
    def _analyze_band_contributions(self, combined_result):
        """
        Analyze the contribution of each band to the final result
        
        Args:
            combined_result: Result dictionary from combine_bands
            
        Returns:
            List of contribution percentages
        """
        weights = combined_result['weights']
        quality_metrics = combined_result['quality_metrics']
        
        contributions = []
        for i, (w, metrics) in enumerate(zip(weights, quality_metrics)):
            contribution = {
                'band_index': i,
                'weight': w,
                'snr_contribution': w * metrics['snr'],
                'entropy_contribution': w * metrics['entropy'],
                'edge_contribution': w * metrics['edge_strength']
            }
            
            if combined_result['band_frequencies'] is not None:
                contribution['frequency_range'] = combined_result['band_frequencies'][i]
            
            contributions.append(contribution)
        
        return contributions
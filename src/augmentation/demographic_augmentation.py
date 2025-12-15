"""
Demographic Augmentation Module

This module implements demographic-aware data augmentation for QoE datasets.
It creates synthetic samples based on user demographic profiles to address
dataset limitations and improve model generalization.
"""

import pandas as pd
import numpy as np
from typing import Dict, Callable, Tuple


class DemographicAugmentation:
    """
    Demographic-aware data augmentation for QoE prediction.
    
    This class implements six distinct user demographic profiles with varying
    QoE sensitivities and applies demographic-specific MOS adjustments to
    create an augmented dataset.
    """
    
    def __init__(self, base_dataset: pd.DataFrame):
        """
        Initialize the demographic augmentation.
        
        Args:
            base_dataset: Original QoE dataset with streaming metrics and MOS scores
        """
        self.df = base_dataset
        self.demographic_profiles = self._define_demographic_profiles()
        
    def _define_demographic_profiles(self) -> Dict:
        """
        Define six demographic profiles with their QoE sensitivity factors.
        
        Returns:
            Dictionary of demographic profiles with sensitivity weights and
            MOS adjustment functions
        """
        profiles = {
            'casual_viewer': {
                'rebuffering_sensitivity': 2.0,
                'quality_sensitivity': 0.6,
                'bitrate_sensitivity': 0.5,
                'consistency_preference': 0.7,
                'mos_adjustment': lambda base_mos, factors: base_mos * (
                    1 - 0.15 * factors['rebuffering_impact'] + 
                    0.05 * factors['quality_boost']
                )
            },
            'quality_enthusiast': {
                'rebuffering_sensitivity': 1.2,
                'quality_sensitivity': 2.5,
                'bitrate_sensitivity': 2.0,
                'consistency_preference': 1.5,
                'mos_adjustment': lambda base_mos, factors: base_mos * (
                    1 + 0.20 * factors['quality_boost'] - 
                    0.25 * factors['quality_variance'] - 
                    0.08 * factors['rebuffering_impact']
                )
            },
            'mobile_user': {
                'rebuffering_sensitivity': 2.5,
                'quality_sensitivity': 0.4,
                'bitrate_sensitivity': 0.3,
                'consistency_preference': 1.8,
                'mos_adjustment': lambda base_mos, factors: base_mos * (
                    1 - 0.20 * factors['rebuffering_impact'] - 
                    0.15 * factors['quality_variance'] + 0.10
                )
            },
            'gamer_sports': {
                'rebuffering_sensitivity': 2.8,
                'quality_sensitivity': 1.5,
                'bitrate_sensitivity': 1.2,
                'consistency_preference': 2.0,
                'mos_adjustment': lambda base_mos, factors: base_mos * (
                    1 - 0.25 * factors['rebuffering_impact'] - 
                    0.20 * factors['quality_variance'] + 
                    0.08 * factors['smoothness']
                )
            },
            'elderly_user': {
                'rebuffering_sensitivity': 2.2,
                'quality_sensitivity': 0.3,
                'bitrate_sensitivity': 0.2,
                'consistency_preference': 1.0,
                'mos_adjustment': lambda base_mos, factors: base_mos * (
                    1 - 0.18 * factors['rebuffering_impact'] + 
                    0.12 * factors['simplicity']
                )
            },
            'professional_critical': {
                'rebuffering_sensitivity': 1.8,
                'quality_sensitivity': 3.0,
                'bitrate_sensitivity': 2.5,
                'consistency_preference': 2.2,
                'mos_adjustment': lambda base_mos, factors: base_mos * (
                    1 + 0.15 * factors['quality_boost'] - 
                    0.35 * factors['quality_variance'] - 
                    0.12 * factors['rebuffering_impact'] - 
                    0.10 * factors['compression_artifacts']
                )
            }
        }
        return profiles
    
    def calculate_impact_factors(self, row: pd.Series) -> Dict[str, float]:
        """
        Calculate normalized impact factors from streaming metrics.
        
        Args:
            row: DataFrame row containing streaming metrics
            
        Returns:
            Dictionary of impact factors (0-1 scale)
        """
        factors = {}
        
        # Rebuffering impact
        factors['rebuffering_impact'] = min(
            row['rebuffering_duration_mean'] / 2.0, 1.0
        )
        
        # Quality boost (VMAF, SSIM)
        vmaf_normalized = row['vmaf_mean'] / 100.0
        ssim_normalized = row['ssim_mean']
        factors['quality_boost'] = (vmaf_normalized + ssim_normalized) / 2.0
        
        # Quality variance (inconsistency penalty)
        vmaf_cv = row['vmaf_std'] / (row['vmaf_mean'] + 1e-5)
        bitrate_cv = row['video_bitrate_std'] / (row['video_bitrate_mean'] + 1e-5)
        factors['quality_variance'] = (vmaf_cv + bitrate_cv) / 2.0
        
        # Smoothness
        factors['smoothness'] = 1.0 - min(factors['quality_variance'], 1.0)
        
        # Simplicity bonus
        factors['simplicity'] = factors['smoothness']
        
        # Compression artifacts (QP-based)
        qp_normalized = min(row['qp_mean'] / 51.0, 1.0)
        factors['compression_artifacts'] = qp_normalized
        
        return factors
    
    def augment(self, noise_std: float = 2.0) -> pd.DataFrame:
        """
        Generate demographic-augmented dataset.
        
        Args:
            noise_std: Standard deviation for individual variation noise
            
        Returns:
            Augmented DataFrame with demographic labels and adjusted MOS scores
        """
        augmented_data = []
        
        for idx, row in self.df.iterrows():
            # Calculate impact factors
            factors = self.calculate_impact_factors(row)
            base_mos = row['mos']
            
            # Generate samples for each demographic
            for demo_name, demo_profile in self.demographic_profiles.items():
                # Adjust MOS based on demographic profile
                adjusted_mos = demo_profile['mos_adjustment'](base_mos, factors)
                
                # Add demographic-specific noise
                noisy_mos = adjusted_mos + np.random.normal(0, noise_std)
                
                # Clip to valid MOS range
                final_mos = np.clip(noisy_mos, 0, 100)
                
                # Create augmented row
                aug_row = row.copy()
                aug_row['mos'] = final_mos
                aug_row['demographic'] = demo_name
                aug_row['original_mos'] = base_mos
                aug_row['mos_adjustment'] = final_mos - base_mos
                
                augmented_data.append(aug_row)
        
        return pd.DataFrame(augmented_data)
    
    def get_profile_info(self) -> pd.DataFrame:
        """
        Get summary information about demographic profiles.
        
        Returns:
            DataFrame with profile characteristics
        """
        profile_info = []
        for name, profile in self.demographic_profiles.items():
            info = {
                'Profile': name,
                'Rebuffering Sensitivity': profile['rebuffering_sensitivity'],
                'Quality Sensitivity': profile['quality_sensitivity'],
                'Bitrate Sensitivity': profile['bitrate_sensitivity'],
                'Consistency Preference': profile['consistency_preference']
            }
            profile_info.append(info)
        
        return pd.DataFrame(profile_info)


def main():
    """Example usage of demographic augmentation."""
    import os
    
    # Load dataset
    df = pd.read_csv("data/combined_dataset.csv")
    
    # Initialize augmentation
    augmentor = DemographicAugmentation(df)
    
    # Display profile information
    print("Demographic Profiles:")
    print(augmentor.get_profile_info())
    
    # Generate augmented dataset
    df_augmented = augmentor.augment()
    
    print(f"\nOriginal dataset: {len(df)} samples")
    print(f"Augmented dataset: {len(df_augmented)} samples")
    print(f"Augmentation factor: {len(df_augmented) / len(df):.1f}x")
    
    # Save augmented dataset
    os.makedirs("data", exist_ok=True)
    df_augmented.to_csv("data/combined_dataset_demographic_augmented.csv", index=False)
    print("\nAugmented dataset saved to data/combined_dataset_demographic_augmented.csv")


if __name__ == "__main__":
    main()

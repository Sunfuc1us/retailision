export interface SKU {sku_id:string;brand:string;product:string;flavor:string;specification:string}
export interface TopKMatch {sku_id:string;score:number}
export interface CandidateInstance {instance_id:number;sku_id:string|null;top1_sku?:string|null;identity_score:number|null;margin?:number|null;top_k?:TopKMatch[];bbox:number[];detection_score?:number;raw_mask_pixels?:number;mask_pixels?:number;bbox_clipped?:boolean;embedding_dim?:number;embedding_norm?:number;mask_path?:string;mask_preview_path?:string;mask_origin?:number[];image_size?:number[];accepted?:boolean;reason?:string}
export interface Prediction {success:boolean;target_found:boolean;reason?:string;sku_id?:string;identity_score?:number;backend?:string;model_id?:string;model_name?:string;instances?:{instance_id:number;sku_id:string;identity_score:number;margin?:number;detection_score?:number;mask_pixels?:number;bbox:number[];mask_path:string}[];timing?:{segmentation_ms?:number;recognition_ms?:number;total_ms:number};candidates?:CandidateInstance[]}
export interface Model {model_name:string;model_version:string;task?:string}

export interface InferenceModel {id:string;name:string;config:string;checkpoint:string|null;recognition_version:string;recognition_checkpoint:string|null;fixture:boolean;backends:string[]}
export interface InferenceModels {default_model:string|null;loaded_model:string|null;models:InferenceModel[]}
